set -x

# Or generate your own using https://github.com/attardi/wikiextractor
aws s3 cp s3://yaroslavvb2/data/wikiextracted.tar .
tar -xf wikiextracted.tar
# Flatten all the files.
find wikiextracted/ -iname 'wiki*' -type f -exec sh -c 'jq -r .text {} > {}.txt' \;
