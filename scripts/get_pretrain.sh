curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76" -o yolor_p6.pt
rm ./cookie
