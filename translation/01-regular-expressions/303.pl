my @links;
while (<>) {
    s/<a\s+href="(.*)"[^>]*>/$1/g;
    s/^(([^:\/\?\#]+):)\/\///;
    s/^([^\/?#:]*).*/$1/;
    s/\n//g;
    push(@links, $_);
}
my %seen;
foreach $link (sort @links)
{
    if (not $seen{$link}) {
        print "$link\n";
        $seen{$link} = 1;
    }
}
