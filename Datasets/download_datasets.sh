if [ -e war_and_peace.txt ]
then
	echo "War and Peace Dataset already found"
else
	curl https://raw.githubusercontent.com/mmcky/nyu-econ-370/master/notebooks/data/book-war-and-peace.txt > war_and_peace.txt
fi
