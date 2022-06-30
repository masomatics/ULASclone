cwd=$(pwd)
mkdir -p /tmp/ulas; cd /tmp
cp -r ${cwd}/* ulas/
rm ssdrl/make_zip.sh
rm ~/reproducing_code.zip
zip -r ~/reproducing_code.zip ulas/* # Make zip file in home directory
rm -r ulas
cd $cwd