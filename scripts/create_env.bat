python3 -m venv env 
echo Activate environment
Call ./env/Scripts/activate.bat
echo update pip
python3 -m pip install --upgrade pip
echo install requirements
python3 -m pip install -e .[dev]