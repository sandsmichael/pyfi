Testing:
    Run from root dir:  python pyfi/test/test.py

Deploy
    Use Setup.py file to build
        pip install setuptools wheel
    Run from root dir
        python setup.py sdist bdist_wheel
    Install wheel
        pip install dist/my_package-0.1.0-py3-none-any.whl


Rebuild Wheel File
    python setup.py sdist bdist_wheel


Note:
git branch -d branch_name
git push origin --delete branch_name

