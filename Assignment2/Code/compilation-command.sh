# Compile
INCLUDE1="./Assignment2/Code/include"
INCLUDE2="./Assignment2/Code/include/tensor"
INCLUDE3="./Assignment2/Code/include/sformat"
INCLUDE4="./Assignment2/Code/include/ann"
INCLUDE5="./Assignment2/Code/demo"
SRC1="./Assignment2/Code/src/ann/"
SRC2="./Assignment2/Code/src/tensor/"
MAIN="./Assignment2/Code/src/program.cpp"

echo "################################################"
echo "# Compilation of the assignment: STARTED #######"
echo "################################################"

g++ -std=c++17 -I "$INCLUDE1" -I "$INCLUDE2" -I "$INCLUDE3" -I "$INCLUDE4" -I "$INCLUDE5" $(find $SRC1 -type f -iregex ".*\.cpp") "$SRC2"/*.cpp "$MAIN"  -o program

echo "################################################"
echo "# Compilation of the assignment: END     #######"
echo "# Binary file output: ./program ################"
echo "################################################"