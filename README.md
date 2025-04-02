# Computer Vision Course
Developed by [Artzai Picon](https://github.com/samtzai)

## Installation in Linux or Linux over Windows (WSL):

1) Install Windows WSL (If you are on Windows)
https://learn.microsoft.com/en-us/windows/wsl/install

    - Requirements:
        - Virtual machines are enabled in your PC (You may need to check you BIOS).
        - Enough free space in your PC
    - Install WSL:
        - Open PowerShell with admin privileges (search it in windows).
        - Enable the necessary features, write:
            dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
            dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
        - Restart your computer.
        - Open PowerShell
        - Set WSL 2 as default version:
            wsl --set-default-version 2
    - Download Ubuntu from Microsoft Store


2) Prepare your ubuntu installation
    - Open Ubuntu terminal (in Windows, you can find it in the start menu after installing Ubuntu from the Microsoft Store)

    - Update and upgrade your system from the terminal:
    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    ```
    - Install python3 and pip3
    ```bash
    sudo apt-get install python3
    sudo apt-get install python3-pip
    ```
    - Install virtualenv
    ```bash
    sudo apt-get install python3-venv
    ```
    - Install git
    ```bash
    sudo apt-get install git
    ```
    - Install make
    ```bash
    sudo apt-get install make
    ```
    - Install Git LFS
    ```bash
    sudo apt-get install git-lfs
    git lfs install
    ```

  
3) Install VS Code in your Windows installation (Not in the WSL)

https://code.visualstudio.com/download or install it from the Microsoft Store

## Installation in Windows (Not in WSL):

1) Install python3 from the Microsoft Store
2) Install VSCode from the Microsoft Store
3) Install the following software:
    - Git: 
    ```bash
    https://git-scm.com/downloads/win
    ```
    - Python: Install it from windows store or download it from the official website
    ```bash
    https://www.python.org/downloads/
    ```
    - Git LFS:
    ```bash
    https://github.com/git-lfs/git-lfs/releases
    ```

## Configure your development stack:

1) Install the following extensions on the extensions tab in VS Code    
    - Recommended extensions in VS Code:
        - WSL
        - Remote Explorer
        - Python
        - Excel Viewer
        - Git Graph
        - Rainbow CSV
        - Ruff
        - vscode-pydata-viewer
        - Markdown Preview Mermaid Support
        - Remote SSH
        - Material Icon Theme

2) Configure git
   
    Configure name and email in git
    ```bash
    git config --global user.name "Your Name"
    git config --global user.email "Your Email"
    ```

    set merge instead of rebase
    ```bash
    git config --global merge.rebase false
    ```

3) Create GitHub account
    - Go to github and create an account.
    - Send your username to the course coordinator to be added to the repository.
    - If not using github classroom:
        - Fork the repository to your account (you will create a new repository with private visibility. Your group will work on the same repository).
        - Give access to the course coordinator to your repository.
    - If using github classroom:
        - Accept the invitation to the classroom


<!-- 4) Create a ssh key and add it to your GitHub account

    - Open a terminal in your Ubuntu installation
    - Create a new ssh key
    ```bash
    ssh-keygen -t rsa -b 4096 -C "" -f ~/.ssh/id_rsa
    ```
    - Add the public key to your GitHub account
    ```bash
    cat ~/.ssh/id_rsa.pub
    ```
    - Copy the output and paste it in your GitHub account (Settings -> SSH and GPG keys -> New SSH key)
 
5) Clone the repository

    - Open a terminal in your Ubuntu installation
    - Create and navigate to the desired directory. For example: 
    ```bash
     mkdir ~/TAIA
     cd ~/TAIA
    ``` 
    - Clone the repository
    ```bash
    git clone https://github.com/your-username/deep_learning_course_torch.git
    ``` -->

## Open the repository in VS Code

1) Open VS Code
2) Connect to the WSL (if you are using it)
    - Open the command palette (Ctrl+Shift+P)
    - Write "Remote-WSL: New Window"
    - Select the WSL installation you want to connect to
3) Clone the repository from github
    - Open the command palette (Ctrl+Shift+P)
    - Write "Git: Clone"
    - select "clone from github". This will help you clone the repository without the need of ssh keys
    - Write the url of the repository: https://github.com/your-username/deep_learning_course_torch
    - Select the folder where you want to clone the repository
3) Open the repository
    - Open the command palette (Ctrl+Shift+P)
    - Write "Open in Folder"
    - Select the folder where you cloned the repository

## Setting up and verifying the environment

1) Open the terminal in VS Code
2) Create the environment:
- Linux or Windows(WSL):
```bash
make create_env
```
- Windows:
```bash
.\scripts\create_env.bat
```
Note: In windows, if large directories are not allowed, change script policy in Powershell with admin rights:

```bash
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" ` -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```


3) Activate the environment:
- Linux or Windows(WSL):
```bash
source env/bin/activate
```
- Windows:
```bash
.\env\Scripts\activate
```

4) Add output folder to git LFS

    We need to add the output folder to git LFS (git storage for large binary files) to avoid pushing the output folder to GitHub.

    - linux or WSL:
    ```bash
    make update_output_dir
    ```
    - windows:
    ```bash
	@echo "Updating output directory..."
	git lfs track outs/**
	git add .gitattributes
    ```



4) see that everything works by running the following command:
- Linux or Windows(WSL):
```bash
make train
``` 
- Windows:
In windows make does not work. Execute the following command:
```bash
python -m src.exercise_01.train
```



## Debug the code in VS Code

1) Open the repository in VS Code
2) Open the "run and debug" section
3) Select the configuration you want to run (e.g. "Python: Current File" or "exercise_01.dataset")
4) Start debugging

Note: In windows, if script running rights are not allowed, change script policy in Powershell with admin rights

```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```


## Submitting the code

When you finish the exercise, you need to submit the code to the course coordinator.

1) Submit the code to the course coordinator by pushing the changes to your repository
```bash
git add .
git commit -m "commit message"
git push
```


To see if the code works, go to the "run and debug" section in VS Code and start running the exercise_01.t01_load_image code
 



