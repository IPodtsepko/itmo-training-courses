// Generated from /home/igor/Documents/courses/translation/parsers-generator/src/main/antlr/org/generator/GrammarParser.g4 by ANTLR 4.9.2

package org.generator;

import org.generator.util.GrammarItem;
import org.generator.util.Import;
import org.generator.util.LexerRule;
import org.generator.util.ParserRule;

import org.generator.util.Alternative;
import org.generator.util.ParserRule.AlternativeItem;
import org.generator.util.Attribute;
import org.generator.util.ParserRule.CodeInsertion;
import org.generator.util.ParserRule.ParserRuleCall;
import org.generator.util.ParserRule.TerminalTransition;

import java.util.List;
import java.util.Map;
import java.util.*;

import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;

import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class GrammarParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.9.2", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		Import=1, Val=2, Def=3, Colon=4, Semicolon=5, Comma=6, OpeningBracket=7, 
		ClosingBracket=8, Identifier=9, Arrow=10, RegularExpression=11, Pipe=12, 
		Whitespaces=13;
	public static final int
		RULE_grammarDescription = 0, RULE_rule = 1, RULE_javaImport = 2, RULE_lexerRuleDeclaration = 3, 
		RULE_parserRuleDeclaration = 4, RULE_attributes = 5, RULE_attribute = 6, 
		RULE_alternatives = 7, RULE_alternative = 8, RULE_unit = 9, RULE_codeInsertion = 10, 
		RULE_token = 11, RULE_maybeCall = 12, RULE_callArguments = 13, RULE_callArgument = 14;
	private static String[] makeRuleNames() {
		return new String[] {
			"grammarDescription", "rule", "javaImport", "lexerRuleDeclaration", "parserRuleDeclaration", 
			"attributes", "attribute", "alternatives", "alternative", "unit", "codeInsertion", 
			"token", "maybeCall", "callArguments", "callArgument"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'import'", "'val'", "'def'", "':'", "';'", "','", "'['", "']'", 
			null, "'->'", null, "'|'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, "Import", "Val", "Def", "Colon", "Semicolon", "Comma", "OpeningBracket", 
			"ClosingBracket", "Identifier", "Arrow", "RegularExpression", "Pipe", 
			"Whitespaces"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "GrammarParser.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public GrammarParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	public static class GrammarDescriptionContext extends ParserRuleContext {
		public List<GrammarItem> values;
		public List<RuleContext> rule() {
			return getRuleContexts(RuleContext.class);
		}
		public RuleContext rule(int i) {
			return getRuleContext(RuleContext.class,i);
		}
		public GrammarDescriptionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_grammarDescription; }
	}

	public final GrammarDescriptionContext grammarDescription() throws RecognitionException {
		GrammarDescriptionContext _localctx = new GrammarDescriptionContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_grammarDescription);
		 ((GrammarDescriptionContext)_localctx).values =  new ArrayList<>(); 
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(31); 
			_errHandler.sync(this);
			_la = _input.LA(1);
			do {
				{
				{
				setState(30);
				rule(_localctx.values);
				}
				}
				setState(33); 
				_errHandler.sync(this);
				_la = _input.LA(1);
			} while ( (((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << Import) | (1L << Val) | (1L << Def))) != 0) );
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class RuleContext extends ParserRuleContext {
		public List<GrammarItem> values;
		public TerminalNode Import() { return getToken(GrammarParser.Import, 0); }
		public JavaImportContext javaImport() {
			return getRuleContext(JavaImportContext.class,0);
		}
		public TerminalNode Semicolon() { return getToken(GrammarParser.Semicolon, 0); }
		public TerminalNode Val() { return getToken(GrammarParser.Val, 0); }
		public LexerRuleDeclarationContext lexerRuleDeclaration() {
			return getRuleContext(LexerRuleDeclarationContext.class,0);
		}
		public TerminalNode Def() { return getToken(GrammarParser.Def, 0); }
		public ParserRuleDeclarationContext parserRuleDeclaration() {
			return getRuleContext(ParserRuleDeclarationContext.class,0);
		}
		public RuleContext(ParserRuleContext parent, int invokingState) { super(parent, invokingState); }
		public RuleContext(ParserRuleContext parent, int invokingState, List<GrammarItem> values) {
			super(parent, invokingState);
			this.values = values;
		}
		@Override public int getRuleIndex() { return RULE_rule; }
	}

	public final RuleContext rule(List<GrammarItem> values) throws RecognitionException {
		RuleContext _localctx = new RuleContext(_ctx, getState(), values);
		enterRule(_localctx, 2, RULE_rule);
		try {
			setState(47);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case Import:
				enterOuterAlt(_localctx, 1);
				{
				setState(35);
				match(Import);
				setState(36);
				javaImport(_localctx.values);
				setState(37);
				match(Semicolon);
				}
				break;
			case Val:
				enterOuterAlt(_localctx, 2);
				{
				setState(39);
				match(Val);
				setState(40);
				lexerRuleDeclaration(_localctx.values);
				setState(41);
				match(Semicolon);
				}
				break;
			case Def:
				enterOuterAlt(_localctx, 3);
				{
				setState(43);
				match(Def);
				setState(44);
				parserRuleDeclaration(_localctx.values);
				setState(45);
				match(Semicolon);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class JavaImportContext extends ParserRuleContext {
		public List<GrammarItem> values;
		public Token RegularExpression;
		public TerminalNode RegularExpression() { return getToken(GrammarParser.RegularExpression, 0); }
		public JavaImportContext(ParserRuleContext parent, int invokingState) { super(parent, invokingState); }
		public JavaImportContext(ParserRuleContext parent, int invokingState, List<GrammarItem> values) {
			super(parent, invokingState);
			this.values = values;
		}
		@Override public int getRuleIndex() { return RULE_javaImport; }
	}

	public final JavaImportContext javaImport(List<GrammarItem> values) throws RecognitionException {
		JavaImportContext _localctx = new JavaImportContext(_ctx, getState(), values);
		enterRule(_localctx, 4, RULE_javaImport);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(49);
			((JavaImportContext)_localctx).RegularExpression = match(RegularExpression);

			            {
			                final var text = (((JavaImportContext)_localctx).RegularExpression!=null?((JavaImportContext)_localctx).RegularExpression.getText():null);
			                _localctx.values.add(new Import(text.substring(1, text.length() - 1)));
			            }
			        
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class LexerRuleDeclarationContext extends ParserRuleContext {
		public List<GrammarItem> values;
		public LexerRule value;
		public Token Identifier;
		public Token RegularExpression;
		public TerminalNode Identifier() { return getToken(GrammarParser.Identifier, 0); }
		public TerminalNode Colon() { return getToken(GrammarParser.Colon, 0); }
		public TerminalNode RegularExpression() { return getToken(GrammarParser.RegularExpression, 0); }
		public LexerRuleDeclarationContext(ParserRuleContext parent, int invokingState) { super(parent, invokingState); }
		public LexerRuleDeclarationContext(ParserRuleContext parent, int invokingState, List<GrammarItem> values) {
			super(parent, invokingState);
			this.values = values;
		}
		@Override public int getRuleIndex() { return RULE_lexerRuleDeclaration; }
	}

	public final LexerRuleDeclarationContext lexerRuleDeclaration(List<GrammarItem> values) throws RecognitionException {
		LexerRuleDeclarationContext _localctx = new LexerRuleDeclarationContext(_ctx, getState(), values);
		enterRule(_localctx, 6, RULE_lexerRuleDeclaration);
		 ((LexerRuleDeclarationContext)_localctx).value =  new LexerRule(); 
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(52);
			((LexerRuleDeclarationContext)_localctx).Identifier = match(Identifier);
			 _localctx.value.setName((((LexerRuleDeclarationContext)_localctx).Identifier!=null?((LexerRuleDeclarationContext)_localctx).Identifier.getText():null)); 
			setState(54);
			match(Colon);
			setState(55);
			((LexerRuleDeclarationContext)_localctx).RegularExpression = match(RegularExpression);

			            {
			                final var text = (((LexerRuleDeclarationContext)_localctx).RegularExpression!=null?((LexerRuleDeclarationContext)_localctx).RegularExpression.getText():null);
			                _localctx.value.setRegularExpression(text.substring(1, text.length() - 1));
			            }
			        

			            _localctx.values.add(_localctx.value);
			        
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ParserRuleDeclarationContext extends ParserRuleContext {
		public List<GrammarItem> values;
		public ParserRule value;
		public AttributesContext attributes;
		public Token Identifier;
		public List<AttributesContext> attributes() {
			return getRuleContexts(AttributesContext.class);
		}
		public AttributesContext attributes(int i) {
			return getRuleContext(AttributesContext.class,i);
		}
		public TerminalNode Identifier() { return getToken(GrammarParser.Identifier, 0); }
		public TerminalNode Arrow() { return getToken(GrammarParser.Arrow, 0); }
		public TerminalNode Colon() { return getToken(GrammarParser.Colon, 0); }
		public List<AlternativesContext> alternatives() {
			return getRuleContexts(AlternativesContext.class);
		}
		public AlternativesContext alternatives(int i) {
			return getRuleContext(AlternativesContext.class,i);
		}
		public ParserRuleDeclarationContext(ParserRuleContext parent, int invokingState) { super(parent, invokingState); }
		public ParserRuleDeclarationContext(ParserRuleContext parent, int invokingState, List<GrammarItem> values) {
			super(parent, invokingState);
			this.values = values;
		}
		@Override public int getRuleIndex() { return RULE_parserRuleDeclaration; }
	}

	public final ParserRuleDeclarationContext parserRuleDeclaration(List<GrammarItem> values) throws RecognitionException {
		ParserRuleDeclarationContext _localctx = new ParserRuleDeclarationContext(_ctx, getState(), values);
		enterRule(_localctx, 8, RULE_parserRuleDeclaration);
		 ((ParserRuleDeclarationContext)_localctx).value =  new ParserRule(); 
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(59);
			((ParserRuleDeclarationContext)_localctx).attributes = attributes();

			          _localctx.value.inherited = ((ParserRuleDeclarationContext)_localctx).attributes.list;
			      
			setState(61);
			((ParserRuleDeclarationContext)_localctx).Identifier = match(Identifier);

			          _localctx.value.name = (((ParserRuleDeclarationContext)_localctx).Identifier!=null?((ParserRuleDeclarationContext)_localctx).Identifier.getText():null);
			      
			setState(63);
			match(Arrow);
			setState(64);
			((ParserRuleDeclarationContext)_localctx).attributes = attributes();

			          _localctx.value.inner = ((ParserRuleDeclarationContext)_localctx).attributes.list;
			      
			setState(66);
			match(Colon);
			setState(68); 
			_errHandler.sync(this);
			_la = _input.LA(1);
			do {
				{
				{
				setState(67);
				alternatives(_localctx.value);
				}
				}
				setState(70); 
				_errHandler.sync(this);
				_la = _input.LA(1);
			} while ( (((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << Identifier) | (1L << RegularExpression) | (1L << Pipe))) != 0) );

			          _localctx.values.add(_localctx.value);
			      
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class AttributesContext extends ParserRuleContext {
		public List<Attribute> list;
		public AttributeContext attribute;
		public TerminalNode OpeningBracket() { return getToken(GrammarParser.OpeningBracket, 0); }
		public TerminalNode ClosingBracket() { return getToken(GrammarParser.ClosingBracket, 0); }
		public List<AttributeContext> attribute() {
			return getRuleContexts(AttributeContext.class);
		}
		public AttributeContext attribute(int i) {
			return getRuleContext(AttributeContext.class,i);
		}
		public List<TerminalNode> Comma() { return getTokens(GrammarParser.Comma); }
		public TerminalNode Comma(int i) {
			return getToken(GrammarParser.Comma, i);
		}
		public AttributesContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_attributes; }
	}

	public final AttributesContext attributes() throws RecognitionException {
		AttributesContext _localctx = new AttributesContext(_ctx, getState());
		enterRule(_localctx, 10, RULE_attributes);
		 ((AttributesContext)_localctx).list =  new ArrayList<>(); 
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(74);
			match(OpeningBracket);
			setState(84);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==RegularExpression) {
				{
				setState(82);
				_errHandler.sync(this);
				switch ( getInterpreter().adaptivePredict(_input,3,_ctx) ) {
				case 1:
					{
					setState(75);
					((AttributesContext)_localctx).attribute = attribute();
					 _localctx.list.add(((AttributesContext)_localctx).attribute.value); 
					}
					break;
				case 2:
					{
					setState(78);
					((AttributesContext)_localctx).attribute = attribute();
					 _localctx.list.add(((AttributesContext)_localctx).attribute.value); 
					setState(80);
					match(Comma);
					}
					break;
				}
				}
				setState(86);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			setState(87);
			match(ClosingBracket);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class AttributeContext extends ParserRuleContext {
		public Attribute value;
		public Token RegularExpression;
		public Token Identifier;
		public TerminalNode RegularExpression() { return getToken(GrammarParser.RegularExpression, 0); }
		public TerminalNode Identifier() { return getToken(GrammarParser.Identifier, 0); }
		public AttributeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_attribute; }
	}

	public final AttributeContext attribute() throws RecognitionException {
		AttributeContext _localctx = new AttributeContext(_ctx, getState());
		enterRule(_localctx, 12, RULE_attribute);
		((AttributeContext)_localctx).value =  new Attribute();
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(89);
			((AttributeContext)_localctx).RegularExpression = match(RegularExpression);

			            {
			                final var text = (((AttributeContext)_localctx).RegularExpression!=null?((AttributeContext)_localctx).RegularExpression.getText():null);
			                _localctx.value.setType(text.substring(1, text.length() - 1));
			            }
			      
			setState(91);
			((AttributeContext)_localctx).Identifier = match(Identifier);

			            _localctx.value.setName((((AttributeContext)_localctx).Identifier!=null?((AttributeContext)_localctx).Identifier.getText():null));
			      
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class AlternativesContext extends ParserRuleContext {
		public ParserRule value;
		public AlternativeContext alternative;
		public List<AlternativeContext> alternative() {
			return getRuleContexts(AlternativeContext.class);
		}
		public AlternativeContext alternative(int i) {
			return getRuleContext(AlternativeContext.class,i);
		}
		public List<TerminalNode> Pipe() { return getTokens(GrammarParser.Pipe); }
		public TerminalNode Pipe(int i) {
			return getToken(GrammarParser.Pipe, i);
		}
		public AlternativesContext(ParserRuleContext parent, int invokingState) { super(parent, invokingState); }
		public AlternativesContext(ParserRuleContext parent, int invokingState, ParserRule value) {
			super(parent, invokingState);
			this.value = value;
		}
		@Override public int getRuleIndex() { return RULE_alternatives; }
	}

	public final AlternativesContext alternatives(ParserRule value) throws RecognitionException {
		AlternativesContext _localctx = new AlternativesContext(_ctx, getState(), value);
		enterRule(_localctx, 14, RULE_alternatives);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(100); 
			_errHandler.sync(this);
			_alt = 1;
			do {
				switch (_alt) {
				case 1:
					{
					{
					setState(95);
					_errHandler.sync(this);
					_la = _input.LA(1);
					if (_la==Pipe) {
						{
						setState(94);
						match(Pipe);
						}
					}

					setState(97);
					((AlternativesContext)_localctx).alternative = alternative();

					            _localctx.value.alternatives.add(((AlternativesContext)_localctx).alternative.line);
					        
					}
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				setState(102); 
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,6,_ctx);
			} while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER );
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class AlternativeContext extends ParserRuleContext {
		public Alternative line;
		public List<UnitContext> unit() {
			return getRuleContexts(UnitContext.class);
		}
		public UnitContext unit(int i) {
			return getRuleContext(UnitContext.class,i);
		}
		public AlternativeContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_alternative; }
	}

	public final AlternativeContext alternative() throws RecognitionException {
		AlternativeContext _localctx = new AlternativeContext(_ctx, getState());
		enterRule(_localctx, 16, RULE_alternative);
		 ((AlternativeContext)_localctx).line =  new Alternative(); 
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(105); 
			_errHandler.sync(this);
			_alt = 1;
			do {
				switch (_alt) {
				case 1:
					{
					{
					setState(104);
					unit(_localctx.line);
					}
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				setState(107); 
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,7,_ctx);
			} while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER );
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class UnitContext extends ParserRuleContext {
		public Alternative units;
		public CodeInsertionContext codeInsertion() {
			return getRuleContext(CodeInsertionContext.class,0);
		}
		public TokenContext token() {
			return getRuleContext(TokenContext.class,0);
		}
		public UnitContext(ParserRuleContext parent, int invokingState) { super(parent, invokingState); }
		public UnitContext(ParserRuleContext parent, int invokingState, Alternative units) {
			super(parent, invokingState);
			this.units = units;
		}
		@Override public int getRuleIndex() { return RULE_unit; }
	}

	public final UnitContext unit(Alternative units) throws RecognitionException {
		UnitContext _localctx = new UnitContext(_ctx, getState(), units);
		enterRule(_localctx, 18, RULE_unit);
		try {
			setState(111);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case RegularExpression:
				enterOuterAlt(_localctx, 1);
				{
				setState(109);
				codeInsertion(_localctx.units);
				}
				break;
			case Identifier:
				enterOuterAlt(_localctx, 2);
				{
				setState(110);
				token(_localctx.units);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class CodeInsertionContext extends ParserRuleContext {
		public Alternative units;
		public Token RegularExpression;
		public TerminalNode RegularExpression() { return getToken(GrammarParser.RegularExpression, 0); }
		public CodeInsertionContext(ParserRuleContext parent, int invokingState) { super(parent, invokingState); }
		public CodeInsertionContext(ParserRuleContext parent, int invokingState, Alternative units) {
			super(parent, invokingState);
			this.units = units;
		}
		@Override public int getRuleIndex() { return RULE_codeInsertion; }
	}

	public final CodeInsertionContext codeInsertion(Alternative units) throws RecognitionException {
		CodeInsertionContext _localctx = new CodeInsertionContext(_ctx, getState(), units);
		enterRule(_localctx, 20, RULE_codeInsertion);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(113);
			((CodeInsertionContext)_localctx).RegularExpression = match(RegularExpression);

			            final var text = (((CodeInsertionContext)_localctx).RegularExpression!=null?((CodeInsertionContext)_localctx).RegularExpression.getText():null);
			            final var code = text.substring(1, text.length() - 1);
			            units.addItem(new CodeInsertion(code));
			        
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class TokenContext extends ParserRuleContext {
		public Alternative units;
		public String name;
		public Token Identifier;
		public MaybeCallContext maybeCall;
		public TerminalNode Identifier() { return getToken(GrammarParser.Identifier, 0); }
		public MaybeCallContext maybeCall() {
			return getRuleContext(MaybeCallContext.class,0);
		}
		public TokenContext(ParserRuleContext parent, int invokingState) { super(parent, invokingState); }
		public TokenContext(ParserRuleContext parent, int invokingState, Alternative units) {
			super(parent, invokingState);
			this.units = units;
		}
		@Override public int getRuleIndex() { return RULE_token; }
	}

	public final TokenContext token(Alternative units) throws RecognitionException {
		TokenContext _localctx = new TokenContext(_ctx, getState(), units);
		enterRule(_localctx, 22, RULE_token);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(116);
			((TokenContext)_localctx).Identifier = match(Identifier);

			            ((TokenContext)_localctx).name =  (((TokenContext)_localctx).Identifier!=null?((TokenContext)_localctx).Identifier.getText():null);
			        
			setState(118);
			((TokenContext)_localctx).maybeCall = maybeCall(_localctx.name);

			            final var parserRules = ((TokenContext)_localctx).maybeCall.parserRules;
			            final var item = parserRules == null ? new TerminalTransition(_localctx.name) : parserRules;
			            units.addItem(item);
			      
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class MaybeCallContext extends ParserRuleContext {
		public String name;
		public ParserRuleCall parserRules;
		public TerminalNode OpeningBracket() { return getToken(GrammarParser.OpeningBracket, 0); }
		public TerminalNode ClosingBracket() { return getToken(GrammarParser.ClosingBracket, 0); }
		public CallArgumentsContext callArguments() {
			return getRuleContext(CallArgumentsContext.class,0);
		}
		public MaybeCallContext(ParserRuleContext parent, int invokingState) { super(parent, invokingState); }
		public MaybeCallContext(ParserRuleContext parent, int invokingState, String name) {
			super(parent, invokingState);
			this.name = name;
		}
		@Override public int getRuleIndex() { return RULE_maybeCall; }
	}

	public final MaybeCallContext maybeCall(String name) throws RecognitionException {
		MaybeCallContext _localctx = new MaybeCallContext(_ctx, getState(), name);
		enterRule(_localctx, 24, RULE_maybeCall);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(127);
			_errHandler.sync(this);
			_la = _input.LA(1);
			if (_la==OpeningBracket) {
				{
				setState(121);
				match(OpeningBracket);

				            ((MaybeCallContext)_localctx).parserRules =  new ParserRuleCall(_localctx.name);
				        
				setState(124);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==Identifier) {
					{
					setState(123);
					callArguments(_localctx.parserRules);
					}
				}

				setState(126);
				match(ClosingBracket);
				}
			}

			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class CallArgumentsContext extends ParserRuleContext {
		public ParserRuleCall parserRule;
		public List<CallArgumentContext> callArgument() {
			return getRuleContexts(CallArgumentContext.class);
		}
		public CallArgumentContext callArgument(int i) {
			return getRuleContext(CallArgumentContext.class,i);
		}
		public List<TerminalNode> Comma() { return getTokens(GrammarParser.Comma); }
		public TerminalNode Comma(int i) {
			return getToken(GrammarParser.Comma, i);
		}
		public CallArgumentsContext(ParserRuleContext parent, int invokingState) { super(parent, invokingState); }
		public CallArgumentsContext(ParserRuleContext parent, int invokingState, ParserRuleCall parserRule) {
			super(parent, invokingState);
			this.parserRule = parserRule;
		}
		@Override public int getRuleIndex() { return RULE_callArguments; }
	}

	public final CallArgumentsContext callArguments(ParserRuleCall parserRule) throws RecognitionException {
		CallArgumentsContext _localctx = new CallArgumentsContext(_ctx, getState(), parserRule);
		enterRule(_localctx, 26, RULE_callArguments);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(134);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,11,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					{
					{
					setState(129);
					callArgument(_localctx.parserRule);
					setState(130);
					match(Comma);
					}
					} 
				}
				setState(136);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,11,_ctx);
			}
			setState(137);
			callArgument(_localctx.parserRule);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class CallArgumentContext extends ParserRuleContext {
		public ParserRuleCall parserRule;
		public Token Identifier;
		public TerminalNode Identifier() { return getToken(GrammarParser.Identifier, 0); }
		public CallArgumentContext(ParserRuleContext parent, int invokingState) { super(parent, invokingState); }
		public CallArgumentContext(ParserRuleContext parent, int invokingState, ParserRuleCall parserRule) {
			super(parent, invokingState);
			this.parserRule = parserRule;
		}
		@Override public int getRuleIndex() { return RULE_callArgument; }
	}

	public final CallArgumentContext callArgument(ParserRuleCall parserRule) throws RecognitionException {
		CallArgumentContext _localctx = new CallArgumentContext(_ctx, getState(), parserRule);
		enterRule(_localctx, 28, RULE_callArgument);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(139);
			((CallArgumentContext)_localctx).Identifier = match(Identifier);
			 _localctx.parserRule.addArgument((((CallArgumentContext)_localctx).Identifier!=null?((CallArgumentContext)_localctx).Identifier.getText():null)); 
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\17\u0091\4\2\t\2"+
		"\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13"+
		"\t\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\3\2\6\2\"\n\2\r\2"+
		"\16\2#\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\5\3\62\n\3\3\4"+
		"\3\4\3\4\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3"+
		"\6\6\6G\n\6\r\6\16\6H\3\6\3\6\3\7\3\7\3\7\3\7\3\7\3\7\3\7\3\7\7\7U\n\7"+
		"\f\7\16\7X\13\7\3\7\3\7\3\b\3\b\3\b\3\b\3\b\3\t\5\tb\n\t\3\t\3\t\3\t\6"+
		"\tg\n\t\r\t\16\th\3\n\6\nl\n\n\r\n\16\nm\3\13\3\13\5\13r\n\13\3\f\3\f"+
		"\3\f\3\r\3\r\3\r\3\r\3\r\3\16\3\16\3\16\5\16\177\n\16\3\16\5\16\u0082"+
		"\n\16\3\17\3\17\3\17\7\17\u0087\n\17\f\17\16\17\u008a\13\17\3\17\3\17"+
		"\3\20\3\20\3\20\3\20\2\2\21\2\4\6\b\n\f\16\20\22\24\26\30\32\34\36\2\2"+
		"\2\u008e\2!\3\2\2\2\4\61\3\2\2\2\6\63\3\2\2\2\b\66\3\2\2\2\n=\3\2\2\2"+
		"\fL\3\2\2\2\16[\3\2\2\2\20f\3\2\2\2\22k\3\2\2\2\24q\3\2\2\2\26s\3\2\2"+
		"\2\30v\3\2\2\2\32\u0081\3\2\2\2\34\u0088\3\2\2\2\36\u008d\3\2\2\2 \"\5"+
		"\4\3\2! \3\2\2\2\"#\3\2\2\2#!\3\2\2\2#$\3\2\2\2$\3\3\2\2\2%&\7\3\2\2&"+
		"\'\5\6\4\2\'(\7\7\2\2(\62\3\2\2\2)*\7\4\2\2*+\5\b\5\2+,\7\7\2\2,\62\3"+
		"\2\2\2-.\7\5\2\2./\5\n\6\2/\60\7\7\2\2\60\62\3\2\2\2\61%\3\2\2\2\61)\3"+
		"\2\2\2\61-\3\2\2\2\62\5\3\2\2\2\63\64\7\r\2\2\64\65\b\4\1\2\65\7\3\2\2"+
		"\2\66\67\7\13\2\2\678\b\5\1\289\7\6\2\29:\7\r\2\2:;\b\5\1\2;<\b\5\1\2"+
		"<\t\3\2\2\2=>\5\f\7\2>?\b\6\1\2?@\7\13\2\2@A\b\6\1\2AB\7\f\2\2BC\5\f\7"+
		"\2CD\b\6\1\2DF\7\6\2\2EG\5\20\t\2FE\3\2\2\2GH\3\2\2\2HF\3\2\2\2HI\3\2"+
		"\2\2IJ\3\2\2\2JK\b\6\1\2K\13\3\2\2\2LV\7\t\2\2MN\5\16\b\2NO\b\7\1\2OU"+
		"\3\2\2\2PQ\5\16\b\2QR\b\7\1\2RS\7\b\2\2SU\3\2\2\2TM\3\2\2\2TP\3\2\2\2"+
		"UX\3\2\2\2VT\3\2\2\2VW\3\2\2\2WY\3\2\2\2XV\3\2\2\2YZ\7\n\2\2Z\r\3\2\2"+
		"\2[\\\7\r\2\2\\]\b\b\1\2]^\7\13\2\2^_\b\b\1\2_\17\3\2\2\2`b\7\16\2\2a"+
		"`\3\2\2\2ab\3\2\2\2bc\3\2\2\2cd\5\22\n\2de\b\t\1\2eg\3\2\2\2fa\3\2\2\2"+
		"gh\3\2\2\2hf\3\2\2\2hi\3\2\2\2i\21\3\2\2\2jl\5\24\13\2kj\3\2\2\2lm\3\2"+
		"\2\2mk\3\2\2\2mn\3\2\2\2n\23\3\2\2\2or\5\26\f\2pr\5\30\r\2qo\3\2\2\2q"+
		"p\3\2\2\2r\25\3\2\2\2st\7\r\2\2tu\b\f\1\2u\27\3\2\2\2vw\7\13\2\2wx\b\r"+
		"\1\2xy\5\32\16\2yz\b\r\1\2z\31\3\2\2\2{|\7\t\2\2|~\b\16\1\2}\177\5\34"+
		"\17\2~}\3\2\2\2~\177\3\2\2\2\177\u0080\3\2\2\2\u0080\u0082\7\n\2\2\u0081"+
		"{\3\2\2\2\u0081\u0082\3\2\2\2\u0082\33\3\2\2\2\u0083\u0084\5\36\20\2\u0084"+
		"\u0085\7\b\2\2\u0085\u0087\3\2\2\2\u0086\u0083\3\2\2\2\u0087\u008a\3\2"+
		"\2\2\u0088\u0086\3\2\2\2\u0088\u0089\3\2\2\2\u0089\u008b\3\2\2\2\u008a"+
		"\u0088\3\2\2\2\u008b\u008c\5\36\20\2\u008c\35\3\2\2\2\u008d\u008e\7\13"+
		"\2\2\u008e\u008f\b\20\1\2\u008f\37\3\2\2\2\16#\61HTVahmq~\u0081\u0088";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}