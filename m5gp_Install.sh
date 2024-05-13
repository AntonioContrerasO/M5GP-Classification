#Installer for m5gp 
echo "***** installing m5gp *****"

echo "******Installing dependencies******"
# Install some dependencies

# remove directory if it exists
if [ -d m5gp ] ; then
     rm -rf m5gp
fi

git clone https://github.com/armandocardenasf/m5gp.git

cd m5gp

echo "****** m5gp Installed******" 