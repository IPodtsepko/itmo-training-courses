use std::collections::HashMap;
use syn::spanned::Spanned;

#[proc_macro_derive(Builder, attributes(builder))]
pub fn derive_builder(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    let name = input.ident;
    let fields = FieldsWrapper::wrap_struct(input.data);
    if let Some(error) = fields.error() {
        let compile_error = error.to_compile_error();
        return quote::quote!(#compile_error).into();
    }
    let field_names = fields.names();
    let field_types = fields.types();
    let setters = fields.setters();
    let builder_default_values = fields.default_values();
    let builder = quote::format_ident!("{}Builder", name);
    let result = quote::quote! {
        pub struct #builder {
            #(#field_names: std::option::Option<#field_types>,)*
        }
        impl #builder {
            #(#setters)*

            pub fn build(&mut self) -> std::result::Result<#name, String> {
                #(if self.#field_names.is_none() {
                    return Err("There are not enough fields to build an object".into());
                })*
                Ok(#name {
                    #(#field_names: self.#field_names.take().unwrap(),)*
                })
            }
        }
        impl #name {
            pub fn builder() -> #builder {
                #builder {
                    #(#field_names: #builder_default_values,)*
                }
            }
        }
    };
    result.into()
}

struct FieldsWrapper {
    fields: Vec<FieldWrapper>,
}

impl FieldsWrapper {
    pub fn wrap_struct(structure: syn::Data) -> Self {
        match structure {
            syn::Data::Struct(structure) => Self::wrap_fields(structure.fields),
            _ => panic!("Builder derive macro support only for structs"),
        }
    }

    pub fn wrap_fields(fields: syn::Fields) -> Self {
        Self {
            fields: match fields {
                syn::Fields::Named(named) => FieldWrapper::wrap(named),
                _ => Vec::new(),
            },
        }
    }

    pub fn error(&self) -> Option<syn::Error> {
        for field in &self.fields {
            if field.error.is_some() {
                return field.error.clone();
            }
        }
        None
    }

    pub fn names(&self) -> Vec<proc_macro2::TokenStream> {
        self.map(|field| field.name())
    }

    pub fn types(&self) -> Vec<proc_macro2::TokenStream> {
        self.map(|field| field.ty())
    }

    pub fn setters(&self) -> Vec<proc_macro2::TokenStream> {
        let mut setters: HashMap<String, proc_macro2::TokenStream> = HashMap::new();
        for field in &self.fields {
            let setter_name = field.setter_name();
            if setters.contains_key(&setter_name) {
                // The field setter is overwritten by the element setters for another field
                continue;
            }
            setters.insert(setter_name, field.setter());
            if field.has_attribute_each() {
                setters.insert(field.attribute_each_value(), field.element_setter());
            }
        }
        setters.into_values().collect()
    }

    pub fn default_values(&self) -> Vec<proc_macro2::TokenStream> {
        self.map(|field| field.default_value())
    }

    pub fn map<F: FnMut(&FieldWrapper) -> proc_macro2::TokenStream>(
        &self,
        mapper: F,
    ) -> Vec<proc_macro2::TokenStream> {
        self.fields.iter().map(mapper).collect()
    }
}

struct FieldWrapper {
    name: proc_macro2::Ident,
    ty: syn::Type,
    underlying: Option<(proc_macro2::Ident, syn::Type)>,
    attribute_each: Option<syn::LitStr>,
    error: Option<syn::Error>,
}

impl FieldWrapper {
    pub fn name(&self) -> proc_macro2::TokenStream {
        let name: &proc_macro2::Ident = &self.name;
        quote::quote!(#name)
    }

    pub fn ty(&self) -> proc_macro2::TokenStream {
        let field_type: &syn::Type = &self.ty;
        quote::quote!(#field_type)
    }

    pub fn default_value(&self) -> proc_macro2::TokenStream {
        let none: proc_macro2::TokenStream = quote::quote!(std::option::Option::None);
        if self.is_type("Option") {
            quote::quote!(std::option::Option::Some(#none))
        } else if self.is_type("Vec") {
            quote::quote!(std::option::Option::Some(std::vec::Vec::new()))
        } else {
            none
        }
    }

    fn is_type(&self, name: &str) -> bool {
        match &self.underlying {
            None => false,
            Some((ident, _)) => ident == name,
        }
    }

    pub fn underlying(&self) -> &syn::Type {
        let (_, underlying): &(proc_macro2::Ident, syn::Type) = self.underlying.as_ref().unwrap();
        underlying
    }

    pub fn setter_name(&self) -> String {
        self.name.to_string()
    }

    pub fn setter(&self) -> proc_macro2::TokenStream {
        let name: &proc_macro2::Ident = &self.name;
        let field_type: &syn::Type = &self.ty;
        if self.is_type("Option") {
            let underlying: &syn::Type = self.underlying();
            quote::quote! {
                pub fn #name(&mut self, #name: #underlying) -> &mut Self {
                    self.#name = std::option::Option::Some(std::option::Option::Some(#name));
                    self
                }
            }
        } else {
            quote::quote! {
                pub fn #name(&mut self, #name: #field_type) -> &mut Self {
                    self.#name = std::option::Option::Some(#name);
                    self
                }
            }
        }
    }

    pub fn has_attribute_each(&self) -> bool {
        self.attribute_each.is_some()
    }

    pub fn attribute_each_value(&self) -> String {
        self.attribute_each
            .as_ref()
            .unwrap()
            .token()
            .to_string()
            .replace('"', "")
    }

    pub fn element_setter(&self) -> proc_macro2::TokenStream {
        let name: &proc_macro2::Ident = &self.name;
        let each: proc_macro2::Ident = quote::format_ident!("{}", self.attribute_each_value());
        let underlying: &syn::Type = self.underlying();
        quote::quote! {
            pub fn #each(&mut self, #each: #underlying) -> &mut Self {
                if let Some(vector) = &mut self.#name {
                    vector.push(#each);
                }
                self
            }
        }
    }

    pub fn wrap(fields: syn::FieldsNamed) -> Vec<Self> {
        fields
            .named
            .iter()
            .map(|field| {
                let name = field.ident.as_ref().unwrap().clone();
                let ty = field.ty.clone();
                let underlying = underlying_of(&ty);
                let attribute_each = attribute_each_of(field);
                let (attribute_each, error) = match attribute_each {
                    Err(error) => (None, Some(error)),
                    Ok(value) => (value, None),
                };
                Self {
                    name,
                    ty,
                    underlying,
                    attribute_each,
                    error,
                }
            })
            .collect()
    }
}

fn underlying_of(ty: &syn::Type) -> Option<(proc_macro2::Ident, syn::Type)> {
    let type_path = match ty {
        syn::Type::Path(type_path) => type_path,
        _ => return None,
    };
    let path_segment = match type_path.path.segments.first() {
        Some(path_segment) => path_segment,
        _ => return None,
    };
    let arguments = match &path_segment.arguments {
        syn::PathArguments::AngleBracketed(arguments) => &arguments.args,
        _ => {
            return None;
        }
    };
    match arguments.first().unwrap() {
        syn::GenericArgument::Type(underlying) => {
            Some((path_segment.ident.clone(), underlying.clone()))
        }

        _ => unreachable!("Not type argument"),
    }
}

fn attribute_each_of(field: &syn::Field) -> syn::Result<Option<syn::LitStr>> {
    let attributes = &field.attrs;
    if attributes.is_empty() {
        return Ok(None);
    }
    let attribute = attributes.first().unwrap();
    let meta = attribute.parse_meta().unwrap();
    let list = match &meta {
        syn::Meta::List(list) => list,
        _ => return Ok(None),
    };
    let nested = match list.nested.first() {
        Some(syn::NestedMeta::Meta(syn::Meta::NameValue(nested))) => nested,
        _ => return Ok(None),
    };
    if nested.path.segments.len() != 1 || nested.path.segments.first().unwrap().ident != "each" {
        return Err(syn::Error::new(
            attribute.tokens.span(),
            "expected `builder(each = \"...\")`",
        ));
    }
    if let syn::Lit::Str(name) = &nested.lit {
        return Ok(Some(name.clone()));
    }
    Err(syn::Error::new(
        nested.lit.span(),
        "expected string literal",
    ))
}
