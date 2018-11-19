!!!
	NOTE: I think we can all use the same virtual environment that is on GitHub. Not sure. Before pulling
	the whole repo, follow the steps to download virtualenv, but do not create a new environment. Try
	to use the one on GitHub first.
!!!


# -------------------------------------- Description -------------------------------------- #

Create a virtualenv. Mine is called 'cs229Project'.
The .bat file called project_env.bat will automatically open the project directory and activate the
virtual env. To use the .bat file you will need to edit the file path. Replace the text of the bat file with:

	@echo off
	cmd /k "cd your/project/directory/path & cs229Project\Scripts\activate"


# ------------------------------------- Initial Setup ------------------------------------- #

The steps I followed to get everything set up:

Upgrade package managers
> conda update conda
> python -m pip install --upgrade pip

Install virtualenv manager
> pip install --user pipenv
-Read the prompt to add the virtualenv exe to your path. Should be in something like 'C:\Users\aristos\AppData\.....'
	-Open 'Edit the Environment Variables
	-Click 'Environment Variables'
	-Under 'User Variables' click 'PATH', click 'edit', add the path from above

Create virtual environment
> cd project_directory
> virtualenv cs229Project (or some other name for your virtual environment)

Activate virtual environment
> cs229Project\Scripts\activate

Install basic packages within environment
> pip install numpy
> pip install pandas
> pip install matplotlib

> pip install tensorflow
> pip install keras