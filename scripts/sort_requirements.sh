# usage: ./scripts/sort_requirements.sh requirements.txt
# run this script from root directory
(cat $1 | sort -k 2) > /tmp/req.txt && mv /tmp/req.txt  $1
