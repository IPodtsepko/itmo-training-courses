$found_empty = 0;
$need_print_empty = 0;
while (<>) {
    if (/^\s*$/) { # is empty line
        $found_empty = 1;
        next;
    }
    if ($need_print_empty and $found_empty) {
        print "\n";
    }
    $found_empty = 0;
    $need_print_empty = 1;
    s/^\s*//; # remove spaces from the begining
    s/\s*$/\n/; # remove spaces from the ending
    s/(\s)+/$1/g;
    print;
}
