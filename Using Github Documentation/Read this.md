Open the html file. The folder contains data that the html file references -- do not delete the folder.

Once the "webpage" loads, here are the steps you should pay attention to.

**STEP 3: Creating a local repository**  
Note that this is only necessary the first time you are initializing a repository. Once a repository (repo) has been initialized, this command is no longer necessary to use (unless for some reason, you want to put the repo somewhere else too). There is no limit to the number of places you can put the repository. You can even put this same repository on the same computer in multiple different places. One limitation: do not put a repo inside a repo.

STEP 4 is only necessary if you are creating a new repository on github (i.e. in the cloud).

###STEP 5,6,8  
These are commands that you will be using a lot, after you have changed things in the repo of course.  
**5: Adding to the Repository**  
The git add command is necessary in that you must tell git which files you are going to commit to the repo. In the example, they specify the files that they have created/edited. This is nice if you are working on many files and only want to commit a portion of them. However, this can be tedioius when you want to add many different files.  If you navigate to the main folder of the repo, you can type "git add ./" which adds all the files in the repo that have been changed, so you dont have to specify each file individually.  
**6: Committing Changes**  
This commits changes to your local repository. Note that it does not commit them to the cloud. Also, make sure you do the -m "message". Github will not let you commit the file unless you have a message (the point is to explain what you are commiting, why, etc).  
**8: Pushing**
Here's a breakdown  
*git - telling the terminal you are using a git command  
*push - put it on the server  
*origin - An alias for the name of the repo on YOUR system. Origin is a default alias given by your system, it is not an actual property of the repo. Changing the alias is discussed later.  
*master - The name of the branch to which you would like to push to. Yes, you can "branch off" of the master and create your own little space for editing code.

###Tips using Github and other useful Commands

**Pulling from Github**  
This is just as important as pushing!!!  
git pull origin master  
This gets all the information that is on github and puts it on your system. IT IS VERY IMPORTANT TO PULL BEFORE YOU PUSH. Github does not allow you to update files unless you have the most recent versions on your computer. Basically, it is preventing you from overwriting changes that other people made to files while you were busy editing those same files!

**Removing Files**  
git rm "file_name"  
This is an important command because "git add ./" does not keep track of files you have deleted.  

**Tracking all additions and removals**  
git add -A ./  
This command is basically a combination of add all files that I added and remove all files I deleted. If you ever delete a file from the repo (which of course, can happen quite frequently), this command is a handy way to track all of the additions and deletions you have made. If you want, you can even use this command instead of git add ./

git status  
This command tells you what changes have been made since your last commit

**Other Tips**  
If multiple people happen to be working on the same repository, it is generally a good idea to have a copy of the code **not** in the repository and make edits to that first. When you are ready to push the code up, I pull from github, then copy my code into the repo, then push it up. If you are editing the code that is directly in the repo, be wary of the following situation.  
You pull from github. You excitedly begin to work on the code within the repo. While you are working, someone else commits changes to the repo, unaware that you are working on code. After some intense work, you finally finish! Yay, congrats. You go to push the code, but github will not allow you to, because you have made edits without pulling the must recent version of the repo! This makes you sad because you must discard the edits before you can pull. 

*Aliases*  
The alias is a system name for your repo. You can view the alias with the following command:  
git remote -v  
Note that this only works when you are inside the repo. If you want to change the alias for any reason, this is the command to do so.  
git remote rename "Current_Alias" "New_Alias"  

### If you are ever in a situation where you need to undo a commit 
These commands will save your life, even with the little things  

**You typed something in the message that you don't want in the message**
git reset --soft HEAD^  
This command will undo the act of committing, but leave all of the files intact (and all of the files that you added).  

**You committed files that you don't want to commit yet**
git reset HEAD^  
This command will undo the act of committing and the act of git add. However, all of the changes you have made to the files will remain the same. 

**You made changes to the files, but you cannot commit because you forgot to pull and someone else changed stuff in the repo**
git reset --hard HEAD^  
This command will reset everything TO THE PREVIOUS COMMIT. Note that this will delete all the file changes you have made in the repo since the last time you pulled, so it is a good idea to have a copy elsewhere before running this command.  
*As a side note, the carot next to HEAD means that it is referencing the previous commit. So for instance, in the master branch, "master" refers to the most recent version, while "master^" refers to the commit just before the most recent version*
