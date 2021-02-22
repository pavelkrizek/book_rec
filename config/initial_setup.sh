# clean up

rm -rf deployment


rm -rf docs

echo "/config.local" >> .dvc/.gitignore
# init git
git init
git remote add origin Url to the app repository.git
git add .
# create and activate the environment
eval "$(conda shell.bash hook)"
conda env create -f environment.yml --quiet
conda activate book_rec
# instal and run pre-commit
pre-commit install
pre-commit run --all-files
# add things to the first commit and push it
git add -u
git commit -m "initial project setup from template"
git push --set-upstream origin master
# install the project locally
python setup.py develop
