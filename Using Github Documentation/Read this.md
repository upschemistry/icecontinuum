Open the html file. The folder contains data that the html file references -- do not delete the folder.

Once the "webpage" loads, here are the steps you should pay attention to.

**STEP 3: Creating a local repository**  
Note that this is only necessary the first time you are initializing a repository. Once a repository (repo) has been initialized, this command is no longer necessary to use (unless for some reason, you want to put the repo somewhere else too). There is no limit to the number of places you can put the repository. You can even put this same repository on the same computer in multiple different places. One limitation: do not put a repo inside a repo.

STEP 4 is only necessary if you are creating a new repository on github (i.e. in the cloud).

**STEP 5,6,8**  
These are commands that you will be using a lot, after you have changed things in the repo of course.  
*5: adding to the repository*  
The git add command is necessary in that you must tell git which files you are going to commit to the repo. In the example, they specify the files that they have created/edited. This is nice if you are working on many files and only want to commit a portion of them. However, this can be tedioius when you want to add many different files.  If you navigate to the main folder of the repo, you can type "git add ./" which adds all the files in the repo that have been changed, so you dont have to specify each file individually.  
*6: Committing Changes*  
This commits changes to your local repository. Note that it does not commit them to the cloud. Also, make sure you do the -m "message". Github will not let you commit the file unless you have a message (the point is to explain what you are commiting, why, etc).  
*8: Pushing*
Here's a breakdown
git - telling the terminal you are using a git command  
push - put it on the server  
origin - An alias for the name of the repo on YOUR system. Origin is a default alias given by your system, it has not an actual property of the repo. Changing the alias is discussed later.  
master - The name of the branch to which you would like to push to. Yes, you can "branch off" of the master and create your own little space for editing code.

**Tips using Github and other useful Commands**

*Pulling from Github*  
This is just as important as pushing!!!  
git pull origin master  
This gets all the information that is on github and puts it on your system. IT IS VERY IMPORTANT TO PULL BEFORE YOU PUSH. Github does not allow you to update files unless you have the most recent versions on your computer. Basically, it is preventing you from overwriting changes that other people made to files while you were busy editing those same files!

*Removing Files*  
git rm "file_name"  
This is an important command because "git add ./" does not keep track of files you have deleted.

*Tips*  
If multiple people happen to be working on the same repository, it is generally a good idea to have a copy of the code **not** in the repository and make edits to that first. When you are ready to push the code up, I pull from github, then 


*Aliases*  
The alias is a system name for your repo. You can view the alias with the following command:  
git remote -v  
Note that this only works when you are inside the repo. If you want to change the alias for any reason, this is the command to do so.  
git remote rename "Current_Alias" "New_Alias"  




git stash
