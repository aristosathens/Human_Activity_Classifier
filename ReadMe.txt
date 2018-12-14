# ------------------------------------- Initial Setup ------------------------------------- #

See the virtual_env_ReadMe for setting up a virtual environment. Important for avoiding
problems with tensorflow versioning.

Raw data can be downloaded from https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring

# ----------------------------------------- Main ------------------------------------------ #

main.py is used for creating a learner object, initializing it, and then training it and/or
monitoring its output. Anything that is specific to your computer's path should go in here,
but hopefully we can standardized our directory layouts.

# ------------------------------------- Parent Class -------------------------------------- #

DataLoader class, found in parent_class.py, is the main class that all other classes inherit
from. It loads data, splits data into training/validation, and assigns indices for accessing
parts of the raw_data.

# -------------------------------------- Subclasses -------------------------------------- #

Each new model or ml technique should be its own class, in its own file, and should inherit
from DataLoader class. For example, neural_net.py contains the DeepLearner() class. Each sub-
class should (probably) overwrite the following methods:
	self.child_init()
	self.train()
	self.loss()
	self.predict()
	self.accuracy()
	self.save()
	self.load()

# -------------------------------------- Enum Types -------------------------------------- #

enum_types.py contains a bunch of enumerated types. Use these for indicating which choice 
you want for a particular parameter. Usually passed in to DataLoader.__init__(), but can 
be used elsewhere. Don't use strings because typos can cause runtime erros that sometimes
don't get caught until your program has been running for hours, crashing and losing all
progress.

# ----------------------------------------- Util ----------------------------------------- #

util.py contains miscellaneous ml and plotting functions.

# ---------------------------------------- Folders --------------------------------------- #

./src/ contains all of the source code.
./data/ contains the raw data. Is gitignored.
./models/ contains any models we have saved for later use.
./output/ contains any plots, images, or anything else to save.
./CS229Project/ contains the virtualenv. Always run project_env.bat before beginning work
	to ensure that we are all working in the same environment. See virtual_env_ReadMe.