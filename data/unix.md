# Unix Commands for DevOps

## System information commands
- `hostname` : displays the hostname of the system
    ```
    $ hostname    
    Eswars-MacBook-Air.local
   ```
- `hostid` : shows the host-id of the system   
- `date`: shows the current data and time of the system in UTC format
    ```
    $ date
    Sun Dec  1 09:17:31 IST 2024
    ```
- `uptime`: Displays the elapsed time duration since the machine logged in
    ```
    $ uptime
    9:23  up 2 days, 20:01, 1 user, load averages: 1.55 1.83 2.15
    ```
- `uname`: Displays the details about the system like unix version
    ```
    $ uname
    Darwin
    ```
- `clear`: Clears the terminal screen
- `history`: lists all the commands executed till now
    ```
    $ history
    1091* find . -type f -iname "*.txt" -exec cat {} ;
    1092* find . -type f -iname "*.txt" -exec cat {} +
    1093* find . -type f -iname "*.py" -exec cat {} +
    1094* telnet localhost:8000
    1095* nc localhost:8000
    1096* nc -v localhost 8000
    1097* nc -v localhost 80
    1098* nc -v localhost 27017
    1099* nc -v localhost 8080
    1100  hostname
    1101  hostid
    1102  brew install hostid
    1103  hostid
    1104  date
    1105  uptime
    1106  uname
    ```
- `sudo`: superuser do - used to execute commands with root user privillages
  ```
  $ sudo su root
  Password:
  sh-3.2#
  ```
- `echo $?`: shows the exit status of the last executed command
  ```
  $ echo $?
  0
  ```
- `shutdown` : shutdown the system
  ```
  $ sudo shutdown -r now # restart the system immediately
  $ sudo shutdown -r +45 # restart the system in next 45 minutes 
  ```
  - `printenv`: displays all the environment variables of the systems
  ```
  $ printenv
  COLORTERM=truecolor
  COMMAND_MODE=unix2003
  HOME=/Users/eswarmaganti
  LC_CTYPE=UTF-8
  LOGNAME=eswarmaganti
  PATH=/opt/anaconda3/bin:/opt/anaconda3/condabin:/Library/Frameworks/Python.framework/Versions/3.10/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin:/Applications/VMware Fusion.app/Contents/Public:/Users/eswarmaganti/Library/Application Support/JetBrains/Toolbox/scripts
  SHELL=/bin/zsh
  SSH_AUTH_SOCK=/private/tmp/com.apple.launchd.gYjlbIQQrl/Listeners
  SSH_SOCKET_DIR=~/.ssh
  TERM=xterm-256color
  TERM_PROGRAM=WarpTerminal
  TERM_PROGRAM_VERSION=v0.2024.11.19.08.02.stable_03
  TMPDIR=/var/folders/ft/7xkx3z2d6fd0l2b1zxlqt7ww0000gn/T/
  USER=eswarmaganti
  WARP_HONOR_PS1=0
  WARP_IS_LOCAL_SHELL_SESSION=1
  WARP_USE_SSH_WRAPPER=1
  XPC_FLAGS=0x0
  XPC_SERVICE_NAME=0
  __CFBundleIdentifier=dev.warp.Warp-Stable
  __CF_USER_TEXT_ENCODING=0x1F5:0:2
  SHLVL=1
  PWD=/Users/eswarmaganti
  OLDPWD=/Users/eswarmaganti
  HOMEBREW_PREFIX=/opt/homebrew
  HOMEBREW_CELLAR=/opt/homebrew/Cellar
  HOMEBREW_REPOSITORY=/opt/homebrew
  INFOPATH=/opt/homebrew/share/info:
  CONDA_EXE=/opt/anaconda3/bin/conda
  _CE_M=
  _CE_CONDA=
  CONDA_PYTHON_EXE=/opt/anaconda3/bin/python
  CONDA_SHLVL=1
  CONDA_PREFIX=/opt/anaconda3
  CONDA_DEFAULT_ENV=base
  CONDA_PROMPT_MODIFIER=(base)
  GSETTINGS_SCHEMA_DIR_CONDA_BACKUP=
  GSETTINGS_SCHEMA_DIR=/opt/anaconda3/share/glib-2.0/schemas
  CONDA_CHANGEPS1=false
  _=/usr/bin/printenv
  ```
- `last`: displays the last logins in the system
  ```
  $ last
  eswarmaganti console                         Thu Nov 28 13:24   still logged in
  reboot time                                Thu Nov 28 13:23
  shutdown time                              Thu Nov 28 13:20
  root       console                         Thu Nov 28 13:19 - shutdown  (00:01)
  eswarmaganti ttys000                         Sun Nov 17 18:12 - 18:12  (00:00)
  eswarmaganti ttys006                         Sun Nov 17 14:05 - 14:05  (00:00)
  eswarmaganti ttys002                         Sun Nov 17 07:19 - 07:19  (00:00)
  eswarmaganti ttys001                         Sun Nov 17 07:19 - 07:19  (00:00)
  eswarmaganti console                         Mon Nov 11 21:35 - 13:19 (16+15:44)
  reboot time                                Mon Nov 11 21:34
  eswarmaganti ttys007                         Mon Nov 11 09:49 - 09:49  (00:00)
  eswarmaganti ttys006                         Mon Nov 11 09:49 - 09:49  (00:00)
  eswarmaganti console                         Fri Nov  8 23:27 - 11:38 (2+12:11)
  reboot time                                Fri Nov  8 23:26
  shutdown time                              Fri Nov  8 11:32
  reboot time                                Fri Nov  8 11:32
  eswarmaganti ttys001                         Fri Nov  8 07:15 - 07:15  (00:00)
  eswarmaganti ttys000                         Fri Nov  8 07:15 - 07:15  (00:00)
  eswarmaganti console                         Fri Nov  8 06:49 - 11:31  (04:42)
  reboot time                                Fri Nov  8 06:47
  shutdown time                              Thu Nov  7 23:42
  eswarmaganti console                         Wed Nov  6 06:50 - 23:42 (1+16:52)
  reboot time                                Wed Nov  6 06:50
  shutdown time                              Tue Nov  5 11:12
  eswarmaganti console                         Sun Nov  3 10:54 - 11:12 (2+00:17)
  reboot time                                Sun Nov  3 10:53
  shutdown time                              Sat Nov  2 06:37
  eswarmaganti console                         Sat Oct 26 18:54 - 06:37 (6+11:43)
  reboot time                                Sat Oct 26 18:54
  shutdown time                              Thu Oct 24 19:52
  eswarmaganti console                         Sun Oct 13 10:48 - 19:52 (11+09:04)
  reboot time                                Sun Oct 13 10:46
  shutdown time                              Sun Oct 13 10:45
  reboot time                                Sun Oct 13 10:45
  shutdown time                              Sat Oct 12 20:27
  eswarmaganti console                         Fri Oct 11 20:33 - 20:27  (23:54)
  reboot time                                Fri Oct 11 20:32
  shutdown time                              Fri Oct 11 20:12
  eswarmaganti console                         Thu Oct 10 05:01 - 20:12 (1+15:11)
  reboot time                                Thu Oct 10 05:00
  eswarmaganti console                         Wed Oct  9 10:56 - 05:00  (18:04)
  reboot time                                Wed Oct  9 10:56
  eswarmaganti ttys000                         Mon Oct  7 15:26 - 15:26  (00:00)
  eswarmaganti ttys001                         Sun Oct  6 04:34 - 04:34  (00:00)
  eswarmaganti ttys000                         Sun Oct  6 04:34 - 04:34  (00:00)
  eswarmaganti console                         Sun Oct  6 04:33 - 09:42 (3+05:09)
  reboot time                                Sun Oct  6 04:31
  shutdown time                              Sun Oct  6 04:30
  root       console                         Sun Oct  6 04:28 - shutdown  (00:01)
  eswarmaganti ttys001                         Sun Sep 29 09:57 - 09:57  (00:00)
  eswarmaganti ttys000                         Sun Sep 29 09:57 - 09:57  (00:00)
  eswarmaganti console                         Fri Sep 27 07:20 - 04:28 (8+21:08)
  reboot time                                Fri Sep 27 07:19
  
  wtmp begins Fri Sep 27 07:19:00 IST 2024
  ```
- `systemctl`: system control - manages system services using systemd
  ```
  $ sudo systemctl start nginx # to start a system service
  $ sudo systemctl enable nginx # to enable a system servcie
  $ sudo systemctl status nginx # displays the status of a system service
  $ sudo systemctl restart nginx # restarts a system service
  ```

# File Commands 
- `touch`: crates files
  ```
  $ touch filename.txt
  $ touch file1.txt file2.txt
  $ # creating multiple files using iteration 
  $ for i {1..10} do; touch file_${i}.txt; done 
  ```
- `cat`: concatenates and displays file context
  ```
  $ cat file.txt # displays files content
  $ cat > file.txt # creates a file with input context interactively
  ```
- `head`: displays first 10 lines of file by default
  ```
  $ head file.txt
  $ head -n 2 file.txt # displays first 2 lines of the file 
  ```
- `tail`: displays last 10 lines of file by default
  ```
  $ tial file.txt # displays last 10 lines of file
  $ tail -n 2 file.txt # displays last 2 lines og file
  ```
- `less`: used to view large files especially log files in a paginated manner
  ```
  $ less file.txt
  ```
- `rm`: remove command
  ```
  $ rm file.txt # deletes a file
  $ rm -r <directory> # deletes a directory
  $ rm -rf <directory> # delete a directory forcefully 
  ```
- `cp` : copy command
  ```
  $ cp <source> <destination>
  $ cp -r dir1 dir2 # copy the contents in dir1 to dir2
  ```

## File permission commands

- `ls` : list and shows the file and its permissions
  ```
  $ ls -l # lists all the files and directories
  total 572864
  drwx------@   8 eswarmaganti  staff       256 Nov  5 07:38 Applications
  drwx------@   9 eswarmaganti  staff       288 Nov 29 11:56 Desktop
  drwxr-xr-x@   5 eswarmaganti  staff       160 Nov 27 16:21 Developer
  drwx------@   5 eswarmaganti  staff       160 Oct 16 15:01 Documents
  drwx------@  33 eswarmaganti  staff      1056 Nov 28 13:18 Downloads
  drwxr-xr-x@   4 eswarmaganti  staff       128 Mar 21  2024 Entertainment
  drwxr-xr-x@   3 eswarmaganti  staff        96 Oct 29 16:46 IdeaProjects
  drwx------@ 100 eswarmaganti  staff      3200 Nov  6 07:21 Library
  drwx------    9 eswarmaganti  staff       288 Jun  7 07:53 Movies
  drwx------+   4 eswarmaganti  staff       128 Mar 24  2024 Music
  drwxr-xr-x@  10 eswarmaganti  staff       320 Oct 12 12:33 Personal
  drwx------+   6 eswarmaganti  staff       192 Aug 25 11:14 Pictures
  drwxr-xr-x@   3 eswarmaganti  staff        96 Apr  3  2024 Postman
  drwxr-xr-x+   5 eswarmaganti  staff       160 May 22  2024 Public
  drwxr-xr-x@   4 eswarmaganti  staff       128 Mar 21  2024 awscli-bundle
  drwxr-xr-x@   4 eswarmaganti  staff       128 Jun  6 10:22 data
  -rw-r--r--@   1 root          staff  94111943 Jun  5 07:01 edb_languagepack_4.app.zip
  -rw-r--r--@   1 root          staff  10265728 Jun  5 07:01 edb_npgsql.app.zip
  -rw-r--r--@   1 root          staff  18420117 Jun  5 07:01 edb_pgagent_pg16.app.zip
  -rw-r--r--@   1 root          staff  34022305 Jun  5 07:01 edb_pgbouncer.app.zip
  -rw-r--r--@   1 root          staff  18090364 Jun  5 07:01 edb_pgjdbc.app.zip
  -rw-r--r--@   1 root          staff  20479917 Jun  5 07:01 edb_psqlodbc.app.zip
  -rw-r--r--    1 eswarmaganti  staff  97893216 May 17  2024 minikube-darwin-amd64
  -rw-r--r--    1 eswarmaganti  staff        72 Oct 24 17:24 test.py
  -rw-r--r--    1 eswarmaganti  staff       446 Oct 24 17:21 test.tf
  ```
  
  ```
  $ ls -ld  # displays the permission of the directory
  drwxr-x---+ 84 eswarmaganti  staff  2688 Dec  1 12:11 .
  ```
  
  ```
  $ ls -la # displays the hidden files
  total 573168
  drwxr-x---+  84 eswarmaganti  staff      2688 Dec  1 12:11 .
  drwxr-xr-x    5 root          admin       160 Nov 28 13:24 ..
  -rw-------    1 eswarmaganti  staff         3 Mar 20  2024 .CFUserTextEncoding
  -rw-r--r--@   1 eswarmaganti  staff     14340 Nov  8 08:39 .DS_Store
  drwx------+  44 eswarmaganti  staff      1408 Dec  1 09:11 .Trash
  drwxr-xr-x    4 eswarmaganti  staff       128 Sep 29 09:56 .anaconda
  drwxr-xr-x@   3 eswarmaganti  staff        96 Apr 17  2024 .android
  drwxr-xr-x@   7 eswarmaganti  staff       224 Aug 27 14:42 .ansible
  drwxr-xr-x@   4 eswarmaganti  staff       128 Oct 12 18:19 .aws
  -rw-------@   1 eswarmaganti  staff     18290 Nov 29 17:14 .bash_history
  -rw-r--r--    1 root          staff       447 Sep 29 09:52 .bash_profile
  drwxr-xr-x@   3 eswarmaganti  staff        96 Oct 28 16:59 .cache
  drwxr-xr-x    4 eswarmaganti  staff       128 Oct  6 04:34 .conda
  -rw-r--r--    1 eswarmaganti  staff        23 Sep 29 09:56 .condarc
  drwxr-xr-x    6 eswarmaganti  staff       192 Oct 15 06:59 .config
  drwxr-xr-x    3 eswarmaganti  staff        96 Sep 29 09:56 .continuum
  drwxr-xr-x@  14 eswarmaganti  staff       448 Nov 20 18:43 .docker
  drwxr-xr-x@   9 eswarmaganti  staff       288 Nov  7 07:15 .eclipse
  -rw-r--r--    1 eswarmaganti  staff       162 Oct 29 15:14 .gitconfig
  drwxr-xr-x    3 eswarmaganti  staff        96 Nov 27 19:19 .gk
  drwxr-xr-x    3 eswarmaganti  staff        96 Apr 17  2024 .groovy
  drwxr-xr-x    4 eswarmaganti  staff       128 Nov 29 08:41 .idlerc
  drwxr-xr-x    4 eswarmaganti  staff       128 Sep 30 05:20 .ipynb_checkpoints
  drwxr-xr-x    3 eswarmaganti  staff        96 Sep 29 09:58 .ipython
  drwxr-xr-x   72 eswarmaganti  staff      2304 Oct 12 20:27 .jenkins
  drwxr-xr-x    3 eswarmaganti  staff        96 Sep 29 09:58 .jupyter
  drwxr-xr-x@   3 eswarmaganti  staff        96 Jul 31 09:37 .k8slens
  drwxr-x---    4 eswarmaganti  staff       128 Nov 25 06:08 .kube
  -rw-------    1 eswarmaganti  staff        20 Dec  1 12:03 .lesshst
  drwxr-xr-x@   3 eswarmaganti  staff        96 Oct 15 06:58 .local
  drwxr-xr-x@   4 eswarmaganti  staff       128 Oct 29 18:27 .m2
  drwxr-xr-x@   3 eswarmaganti  staff        96 Oct 27 05:37 .microk8s
  drwxr-xr-x   19 eswarmaganti  staff       608 Nov 13 07:39 .minikube
  drwx------@   4 eswarmaganti  staff       128 Mar 26  2024 .mongodb
  -rw-------@   1 eswarmaganti  staff       153 Jun  5 09:15 .node_repl_history
  drwxr-xr-x@   6 eswarmaganti  staff       192 Jun  5 06:54 .npm
  drwxr-xr-x@   8 eswarmaganti  staff       256 Nov 11 08:20 .p2
  drwx------@  16 eswarmaganti  staff       512 Nov 17 15:20 .pgadmin
  -rw-------@   1 eswarmaganti  staff      1684 Nov 11 09:52 .psql_history
  -rw-------    1 eswarmaganti  staff      3398 Nov 27 20:53 .python_history
  drwxr-xr-x@   8 eswarmaganti  staff       256 Jul 16 15:33 .redis-insight
  -rw-------    1 eswarmaganti  staff        35 Jun  5 08:31 .rediscli_history
  drwxr-xr-x@   6 eswarmaganti  staff       192 Apr  5  2024 .rest-client
  drwxr-xr-x    3 eswarmaganti  staff        96 Aug 23 17:16 .sonar
  drwxr-xr-x@  14 eswarmaganti  staff       448 Nov 19 20:28 .ssh
  -rw-r--r--    1 root          staff       281 Sep 29 09:52 .tcshrc
  drwxr-xr-x@   4 eswarmaganti  staff       128 Nov 27 19:53 .terraform.d
  drwxr-xr-x    4 eswarmaganti  staff       128 Jun  5 08:21 .th-client
  -rw-------    1 eswarmaganti  staff     27267 Nov 29 09:07 .viminfo
  drwxr-xr-x    3 eswarmaganti  staff        96 Oct 15 05:24 .vs-kubernetes
  drwxr-xr-x@   5 eswarmaganti  staff       160 Mar 20  2024 .vscode
  drwxr-xr-x    3 eswarmaganti  staff        96 Oct 23 11:38 .warp
  -rw-r--r--    1 root          staff       635 Sep 29 09:52 .xonshrc
  -rw-r--r--@   1 eswarmaganti  staff       495 Sep 12 07:23 .zprofile
  -rw-r--r--@   1 eswarmaganti  staff        43 Mar 20  2024 .zprofile.bak
  -rw-r--r--@   1 eswarmaganti  staff       329 Apr  2  2024 .zprofile.pysave
  -rw-------    1 eswarmaganti  staff     30619 Dec  1 12:11 .zsh_history
  drwx------   18 eswarmaganti  staff       576 Nov 17 18:12 .zsh_sessions
  -rw-r--r--@   1 eswarmaganti  staff       460 Sep 29 09:52 .zshrc
  drwx------@   8 eswarmaganti  staff       256 Nov  5 07:38 Applications
  drwx------@   9 eswarmaganti  staff       288 Nov 29 11:56 Desktop
  drwxr-xr-x@   5 eswarmaganti  staff       160 Nov 27 16:21 Developer
  drwx------@   5 eswarmaganti  staff       160 Oct 16 15:01 Documents
  drwx------@  33 eswarmaganti  staff      1056 Nov 28 13:18 Downloads
  drwxr-xr-x@   4 eswarmaganti  staff       128 Mar 21  2024 Entertainment
  drwxr-xr-x@   3 eswarmaganti  staff        96 Oct 29 16:46 IdeaProjects
  drwx------@ 100 eswarmaganti  staff      3200 Nov  6 07:21 Library
  drwx------    9 eswarmaganti  staff       288 Jun  7 07:53 Movies
  drwx------+   4 eswarmaganti  staff       128 Mar 24  2024 Music
  drwxr-xr-x@  10 eswarmaganti  staff       320 Oct 12 12:33 Personal
  drwx------+   6 eswarmaganti  staff       192 Aug 25 11:14 Pictures
  drwxr-xr-x@   3 eswarmaganti  staff        96 Apr  3  2024 Postman
  drwxr-xr-x+   5 eswarmaganti  staff       160 May 22  2024 Public
  drwxr-xr-x@   4 eswarmaganti  staff       128 Mar 21  2024 awscli-bundle
  drwxr-xr-x@   4 eswarmaganti  staff       128 Jun  6 10:22 data
  -rw-r--r--@   1 root          staff  94111943 Jun  5 07:01 edb_languagepack_4.app.zip
  -rw-r--r--@   1 root          staff  10265728 Jun  5 07:01 edb_npgsql.app.zip
  -rw-r--r--@   1 root          staff  18420117 Jun  5 07:01 edb_pgagent_pg16.app.zip
  -rw-r--r--@   1 root          staff  34022305 Jun  5 07:01 edb_pgbouncer.app.zip
  -rw-r--r--@   1 root          staff  18090364 Jun  5 07:01 edb_pgjdbc.app.zip
  -rw-r--r--@   1 root          staff  20479917 Jun  5 07:01 edb_psqlodbc.app.zip
  -rw-r--r--    1 eswarmaganti  staff  97893216 May 17  2024 minikube-darwin-amd64
  -rw-r--r--    1 eswarmaganti  staff        72 Oct 24 17:24 test.py
  -rw-r--r--    1 eswarmaganti  staff       446 Oct 24 17:21 test.tf
  ```
  
- `chmod` : change the permissions of files and directories
  ```
  # syntax: chmod <mode> <fileName>
  ```
  ```
  $ chmod 700 file.txt # allow read(4), write(2), execute(1) permissions for user
  $ chmod -R 777 <directory> # change the direcory permissions (all permissions to user, group and others) recursively
  ```
- `chown`: change the ownership of files / directories
  ```
  syntax: chown <user>:<group> <file/directory> 
  ```
  ```
  $ chown ubuntu: /opt/SP/automation/script.py # update the ownership of the specific file
  $ chown -R ubuntu: /opt/SP/automation # updates the all the objects ownership in the specified dorectory
  ```
- `chgrp`: update the group name of the file / directory
  ```
  syntax: chgrp <group_name> <file/direcotry>
  ```
  ```
  $ chgrp wheel /opt/SP/automation # update the group name to wheel
  ```
- `getfacl` : shows the file/directory access control list
  ```
  $ getfacl file.txt 
  # file: file.txt
  # owner: ubuntu
  # group: ubuntu
  user::rw-
  group::rw-
  other::r--
  ```
- `setfacl` : manages the access control list permissions for file / directory 
  ```
  $ setfacl -m u:test:rwx file.txt  # modifies the current acl of the file to test user

  $ getfacl file.txt 
  # file: file.txt
  # owner: ubuntu
  # group: ubuntu
  user::rw-
  user:test:rwx
  group::rw-
  mask::rwx
  other::r--
  ```
  ```
  $ setfacl -m g:testgrp:rw file.txt
  
  $ getfacl file.txt
  # file: file.txt
  # owner: ubuntu
  # group: ubuntu
  user::rw-
  user:test:rwx
  group::rw-
  group:testgrp:rw-
  mask::rwx
  other::r--
  ```

# User management commands
- `ac` : Displays the total connect time for all users or a specific user
  ```
  $ ac ubuntu # displays ligin time for specific user
	 total       11.97
  ```
  ```
  $ ac -d ubuntu # displays login time for each day
    Nov  3	total        9.20
    Nov  4	total        0.12
    Dec  1	total        2.60
    Today	total        0.05
  ```
  ```
  $ ac -pd ubuntu # displays login time for current day
	ubuntu                               9.20
    Nov  3	total        9.20
    ubuntu                               0.12
    Nov  4	total        0.12
    ubuntu                               2.60
    Dec  1	total        2.60
    ubuntu                               0.05
    Today	total        0.05
  ```

- `useradd` : creates an user account
  ```
  $ sudo useradd readonly # creates the user without home and mail spool directories
  
  $ sudo useradd readonlyuser # creates the user with home and mail spool directories
  ```

- `passwd` : system generates the password for the user and stores it in the /etc/shadow file
  ```
  $ sudo passwd readonly
  New password: 
  Retype new password:
  passwd: password updated successfully
  ```

- `userdel` : deletes a user account
  ```
  $ sudo userdel readonly # deletes a specific user without deleting home and mail spool dirs
  
  $ sudo userdel -r readonlyuser  # deletes the home and mail spool directories
  userdel: readonlyuser mail spool (/var/mail/readonlyuser) not found
  ```

- `/etc/passwd` : stores the information about all users
  ```
  $ cat /etc/passwd
  cat /etc/passwd
  root:x:0:0:root:/root:/bin/bash
  daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
  bin:x:2:2:bin:/bin:/usr/sbin/nologin
  sys:x:3:3:sys:/dev:/usr/sbin/nologin
  sync:x:4:65534:sync:/bin:/bin/sync
  games:x:5:60:games:/usr/games:/usr/sbin/nologin
  man:x:6:12:man:/var/cache/man:/usr/sbin/nologin
  lp:x:7:7:lp:/var/spool/lpd:/usr/sbin/nologin
  mail:x:8:8:mail:/var/mail:/usr/sbin/nologin
  news:x:9:9:news:/var/spool/news:/usr/sbin/nologin
  uucp:x:10:10:uucp:/var/spool/uucp:/usr/sbin/nologin
  proxy:x:13:13:proxy:/bin:/usr/sbin/nologin
  www-data:x:33:33:www-data:/var/www:/usr/sbin/nologin
  backup:x:34:34:backup:/var/backups:/usr/sbin/nologin
  list:x:38:38:Mailing List Manager:/var/list:/usr/sbin/nologin
  irc:x:39:39:ircd:/run/ircd:/usr/sbin/nologin
  _apt:x:42:65534::/nonexistent:/usr/sbin/nologin
  nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin
  systemd-network:x:998:998:systemd Network Management:/:/usr/sbin/nologin
  systemd-timesync:x:996:996:systemd Time Synchronization:/:/usr/sbin/nologin
  dhcpcd:x:100:65534:DHCP Client Daemon,,,:/usr/lib/dhcpcd:/bin/false
  messagebus:x:101:101::/nonexistent:/usr/sbin/nologin
  syslog:x:102:102::/nonexistent:/usr/sbin/nologin
  systemd-resolve:x:991:991:systemd Resolver:/:/usr/sbin/nologin
  uuidd:x:103:103::/run/uuidd:/usr/sbin/nologin
  tss:x:104:104:TPM software stack,,,:/var/lib/tpm:/bin/false
  sshd:x:105:65534::/run/sshd:/usr/sbin/nologin
  pollinate:x:106:1::/var/cache/pollinate:/bin/false
  tcpdump:x:107:108::/nonexistent:/usr/sbin/nologin
  landscape:x:108:109::/var/lib/landscape:/usr/sbin/nologin
  fwupd-refresh:x:990:990:Firmware update daemon:/var/lib/fwupd:/usr/sbin/nologin
  polkitd:x:989:989:User for polkitd:/:/usr/sbin/nologin
  ec2-instance-connect:x:109:65534::/nonexistent:/usr/sbin/nologin
  _chrony:x:110:112:Chrony daemon,,,:/var/lib/chrony:/usr/sbin/nologin
  ubuntu:x:1000:1000:Ubuntu:/home/ubuntu:/bin/bash
  test:x:1001:1001::/home/test:/bin/sh
  
  $ awk '{FS=":"}{print $1}' /etc/passwd  # printing only the usernames using awk command
  root:x:0:0:root:/root:/bin/bash
  daemon
  bin
  sys
  sync
  games
  man
  lp
  mail
  news
  uucp
  proxy
  www-data
  backup
  list
  irc
  _apt
  nobody
  systemd-network
  systemd-timesync
  dhcpcd
  messagebus
  syslog
  systemd-resolve
  uuidd
  tss
  sshd
  pollinate
  tcpdump
  landscape
  fwupd-refresh
  polkitd
  ec2-instance-connect
  _chrony
  ubuntu
  test
  ```

- `/etc/shadow` : stores all the passwords for user accounts in encrypted format
  ```
  $ sudo tail -n 5 /etc/shadow
  sudo tail -n 5 /etc/shadow
  polkitd:!*:19993::::::
  ec2-instance-connect:!:19993::::::
  _chrony:!:19993::::::
  ubuntu:$y$j9T$laLB6VjRSAHzaaJ1fG9sY1$yEo2pwrRQOFKAIQM2AfYliRCRkcn64P7kxuUQ1uUPwB:20059:0:99999:7:::
  test:$y$j9T$ac8qM.AtQsG2Ky6JR/DjL0$MeJgRzFPGtqQL2sBwhlOX/2cEgcXU3tcV10OdPxUAZ5:20059:0:99999:7:::
  ```

- `su` : substitute user 
  ```
  $ su test
  su test
  Password:
  $ exit
  ```

- `usermod` : modify user 
  ```
  # append the user to a group with out removing the user from other groups
  $ sudo usermod -aG developer ubuntu
  ```

## Group management commands

- `groupadd` : creates a new group
  ```
  $ sudo groupadd developers
  ```
- `groupdel` : delete an existing group
  ```
  $ sudo groupdel developers
  ```
- `/etc/group` : stores all the groups
  ```
  $ sudo cat /etc/group
  root:x:0:
  daemon:x:1:
  bin:x:2:
  sys:x:3:
  adm:x:4:syslog,ubuntu
  tty:x:5:
  disk:x:6:
  lp:x:7:
  mail:x:8:
  news:x:9:
  uucp:x:10:
  man:x:12:
  proxy:x:13:
  kmem:x:15:
  dialout:x:20:
  fax:x:21:
  voice:x:22:
  cdrom:x:24:ubuntu
  floppy:x:25:
  tape:x:26:
  sudo:x:27:ubuntu
  audio:x:29:
  dip:x:30:ubuntu
  www-data:x:33:
  backup:x:34:
  operator:x:37:
  list:x:38:
  irc:x:39:
  src:x:40:
  shadow:x:42:
  utmp:x:43:
  video:x:44:
  sasl:x:45:
  plugdev:x:46:
  staff:x:50:
  games:x:60:
  users:x:100:
  nogroup:x:65534:
  systemd-journal:x:999:
  systemd-network:x:998:
  crontab:x:997:
  systemd-timesync:x:996:
  input:x:995:
  sgx:x:994:
  kvm:x:993:
  render:x:992:
  messagebus:x:101:
  syslog:x:102:
  systemd-resolve:x:991:
  uuidd:x:103:
  tss:x:104:
  lxd:x:105:ubuntu
  _ssh:x:106:
  rdma:x:107:
  tcpdump:x:108:
  landscape:x:109:
  fwupd-refresh:x:990:
  polkitd:x:989:
  admin:x:110:
  netdev:x:111:
  _chrony:x:112:
  ubuntu:x:1000:
  test:x:1001:
  testgrp:x:1002:
  developers:x:1003:
  ```
- `gpassword` : creates a password for the group
  ```
  $ sudo gpasswd developers
  Changing the password for group developers
  New Password:
  Re-enter new password:
  ```
  
  ```
  # adds a user to the group
  $ sudo gpasswd -a ubuntu developers
  Adding user ubuntu to group developers
  
  # removing a user from a group
  $ sudo gpasswd -d ubuntu developers
  Removing user ubuntu from group developers
  
  # adds multiple users to a group
  $ sudo gpasswd -M ubuntu,test developers
  ```

## Search Commands
- `find` : search for files / directories based on their names, creation time, modified time, permissions, and content using other commands

  ```
  # search single file by name
  $ find / -iname "foo.txt" 2>/dev/null
  /opt/SP/foo.txt
  /home/ubuntu/foo.txt
  ```
  
  ```
  # search single file by approximate name
  $ find / -iname "foo*.txt" 2>/dev/null
  /opt/SP/foo.txt
  /home/ubuntu/foo.txt
  /home/ubuntu/foo2.txt
  ```

  ```
  # Finds everything (the -ls option used to list all the files recursivley inside the specified directory )
  $ find /opt/SP/student-mgmt-flask-app/models -ls
   271189      4 drwxr-xr-x   3 ubuntu   root         4096 Nov  3 06:21 /opt/SP/student-mgmt-flask-app/models
   303455      4 drwxr-xr-x   2 root     root         4096 Nov  3 06:21 /opt/SP/student-mgmt-flask-app/models/__pycache__
   303485      4 -rw-r--r--   1 root     root         1481 Nov  3 06:21 /opt/SP/student-mgmt-flask-app/models/__pycache__/__init__.cpython-312.pyc
   271190      4 -rw-r--r--   1 ubuntu   root          555 Nov  3 06:19 /opt/SP/student-mgmt-flask-app/models/__init__.py
  ```
  
  ```
  # find the files by content (using -exec option with grep command to find a files by content)
  $ find / -iname 'foo*.txt' -exec grep -Hi "testing" {} \; 2>/dev/null
  /opt/SP/foo.txt:This is testing food file
  /home/ubuntu/foo.txt:This is a foo file, created for testing purpose
  /home/ubuntu/foo2.txt:testing foo
  ```
  
  ```
  # find files by type
  $ find / -type f -name "*.log" 2>/dev/null
  /usr/share/doc/python3.12/pybench.log
  /run/cloud-init/cloud-init-generator.log
  /run/cloud-init/ds-identify.log
  /var/log/auth.log
  /var/log/dpkg.log
  /var/log/alternatives.log
  /var/log/cloud-init-output.log
  /var/log/apt/history.log
  /var/log/apt/term.log
  /var/log/kern.log
  /var/log/nginx/error.log
  /var/log/nginx/access.log
  /var/log/cloud-init.log
  /var/log/landscape/sysinfo.log
  /var/log/apport.log
  /var/log/unattended-upgrades/unattended-upgrades-shutdown.log
  /var/log/unattended-upgrades/unattended-upgrades-dpkg.log
  /var/log/unattended-upgrades/unattended-upgrades.log
  ```
  
  ```
  # Listing all the directories
  $ find /var/log -type d -ls 2>/dev/null
    81280      4 drwxrwxr-x  13 root     syslog       4096 Dec  2 01:18 /var/log
    81287      4 drwxr-xr-x   2 root     root         4096 Sep  5 19:11 /var/log/dist-upgrade
   262267      4 drwxr-xr-x   2 root     root         4096 Dec  2 01:18 /var/log/account
   262351      4 drwxr-x---   2 _chrony  _chrony      4096 Nov  3 05:34 /var/log/chrony
   262401      4 drwx------   3 root     root         4096 Nov  3 05:34 /var/log/amazon
    81285      4 drwxr-sr-x   3 root     systemd-journal     4096 Nov  3 05:34 /var/log/journal
   262145      4 drwxr-sr-x   2 root     systemd-journal     4096 Dec  1 06:59 /var/log/journal/ec2e42256ca842f991879decfa26d08c
    81286      4 drwxr-xr-x   2 root     root                4096 Dec  2 01:18 /var/log/apt
   269683      4 drwxr-xr-x   2 root     adm                 4096 Dec  2 00:00 /var/log/nginx
   262150      4 drwx------   2 root     root                4096 Nov  3 05:34 /var/log/private
    81289      4 drwxr-xr-x   2 landscape landscape           4096 Nov  3 05:35 /var/log/landscape
    81290      4 drwxr-x---   2 root      adm                 4096 Dec  1 06:59 /var/log/unattended-upgrades
    81288      4 drwxr-xr-x   2 root      root                4096 Dec  2 00:07 /var/log/sysstat
  ```
  ```
  # Limiting the results using -maxdepth option
  $ find /home -maxdepth 3 -type d -ls
     1728      4 drwxr-xr-x   3 root     root         4096 Dec  2 01:35 /home
   262339      4 drwxr-x---   6 ubuntu   ubuntu       4096 Dec  2 03:29 /home/ubuntu
   267907      4 drwxrwxr-x   3 ubuntu   ubuntu       4096 Nov  3 05:41 /home/ubuntu/.local
   267908      4 drwxrwxr-x   3 ubuntu   ubuntu       4096 Nov  3 05:41 /home/ubuntu/.local/share
   269766      4 drwx------   3 ubuntu   ubuntu       4096 Nov  3 06:16 /home/ubuntu/.ansible
   269774      4 drwx------   2 ubuntu   ubuntu       4096 Nov  3 06:21 /home/ubuntu/.ansible/tmp
   262344      4 drwx------   2 ubuntu   ubuntu       4096 Nov  3 05:34 /home/ubuntu/.ssh
   262354      4 drwx------   3 ubuntu   ubuntu       4096 Nov  3 05:41 /home/ubuntu/.cache
   269494      4 drwxrwxr-x   3 ubuntu   ubuntu       4096 Nov  3 05:41 /home/ubuntu/.cache/pip
  ```
  ```
  # finding all the empty files
  $ find /home -type f -empty
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any/pip/_vendor/urllib3/contrib/_securetransport/__init__.py
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any/pip/_vendor/urllib3/contrib/__init__.py
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any/pip/_vendor/urllib3/packages/backports/__init__.py
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any/pip/_vendor/urllib3/packages/__init__.py
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any/pip/_vendor/resolvelib/compat/__init__.py
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any/pip/_vendor/chardet/cli/__init__.py
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any/pip/_vendor/chardet/metadata/__init__.py
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any/pip/_internal/resolution/legacy/__init__.py
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any/pip/_internal/resolution/resolvelib/__init__.py
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any/pip/_internal/resolution/__init__.py
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any/pip/_internal/utils/__init__.py
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any/pip/_internal/operations/build/__init__.py
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any/pip/_internal/operations/__init__.py
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any/pip-24.0.virtualenv
  /home/ubuntu/.local/share/virtualenv/wheel/3.12/image/1/CopyPipInstall/pip-24.0-py3-none-any.lock
  /home/ubuntu/.sudo_as_admin_successful
  /home/ubuntu/.cache/motd.legal-displayed
  ```
  ```
  # finding the files based on the modified time
  $ find /home -type f -mtime -7
  /home/ubuntu/foo.txt
  /home/ubuntu/file.txt
  /home/ubuntu/.bash_history
  /home/ubuntu/foo2.txt
  ```
  
  ```
  # Search paths using find command with -path option
  $ find / -type d -path "/opt/SP/*models" 2>/dev/null
  /opt/SP/student-mgmt-flask-app/venv/lib/python3.12/site-packages/pip/_internal/models
  /opt/SP/student-mgmt-flask-app/models
  ```

- `locate` : locate is faster for finding files by name due to its pre-built database.
  ```
  # syntax:
  $ locate <fileName/dirName>
  ```

## Hardware information commands
- `free` : displays system memory information 
  ```
  $ free -h
               total        used        free      shared  buff/cache   available
  Mem:           957Mi       512Mi       166Mi       892Ki       471Mi       444Mi
  Swap:             0B          0B          0B
  ```
- `df` : displays disk usage space for mounted filesystems
  ```
  $ df -h
  Filesystem      Size  Used Avail Use% Mounted on
  /dev/root       6.8G  2.7G  4.1G  41% /
  tmpfs           479M     0  479M   0% /dev/shm
  tmpfs           192M  876K  191M   1% /run
  tmpfs           5.0M     0  5.0M   0% /run/lock
  /dev/xvda16     881M  133M  687M  17% /boot
  /dev/xvda15     105M  6.1M   99M   6% /boot/efi
  tmpfs            96M   12K   96M   1% /run/user/1000
  ```
- `du` : displays disk usage information
  ```
  # displays the size of individual files
  $ du /opt -h
  8.0K	/opt/SP/student-mgmt-flask-app/templates/components
  36K	/opt/SP/student-mgmt-flask-app/templates
  8.0K	/opt/SP/student-mgmt-flask-app/static/styles
  12K	/opt/SP/student-mgmt-flask-app/static
  4.0K	/opt/SP/student-mgmt-flask-app/.git/branches
  4.0K	/opt/SP/student-mgmt-flask-app/.git/refs/tags
  16K	/opt/SP/student-mgmt-flask-app/.git/refs/remotes/origin
  20K	/opt/SP/student-mgmt-flask-app/.git/refs/remotes
  8.0K	/opt/SP/student-mgmt-flask-app/.git/refs/heads
  36K	/opt/SP/student-mgmt-flask-app/.git/refs
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/57
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/8c
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/5e
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/56
  4.0K	/opt/SP/student-mgmt-flask-app/.git/objects/info
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/a0
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/48
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/5b
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/ff
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/f0
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/7a
  28K	/opt/SP/student-mgmt-flask-app/.git/objects/pack
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/71
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/07
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/13
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/31
  8.0K	/opt/SP/student-mgmt-flask-app/.git/objects/49
  156K	/opt/SP/student-mgmt-flask-app/.git/objects
  8.0K	/opt/SP/student-mgmt-flask-app/.git/info
  16K	/opt/SP/student-mgmt-flask-app/.git/logs/refs/remotes/origin
  20K	/opt/SP/student-mgmt-flask-app/.git/logs/refs/remotes
  8.0K	/opt/SP/student-mgmt-flask-app/.git/logs/refs/heads
  32K	/opt/SP/student-mgmt-flask-app/.git/logs/refs
  ```
  ```
  # displays the total size of the directory
  $ du -sh /opt
  46M	/opt
  ```

- `lscpu` : displays the system cpu hardware information
  ```
  $ lscpu
  Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          46 bits physical, 48 bits virtual
  Byte Order:             Little Endian
  CPU(s):                   1
  On-line CPU(s) list:    0
  Vendor ID:                GenuineIntel
  Model name:             Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
  CPU family:           6
  Model:                79
  Thread(s) per core:   1
  Core(s) per socket:   1
  Socket(s):            1
  Stepping:             1
  BogoMIPS:             4599.99
  Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse
  2 ht syscall nx rdtscp lm constant_tsc rep_good nopl xtopology cpuid tsc_known_freq pni pclmulqd
  q ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rd
  rand hypervisor lahf_lm abm pti fsgsbase bmi1 avx2 smep bmi2 erms invpcid xsaveopt
  Virtualization features:  
  Hypervisor vendor:      Xen
  Virtualization type:    full
  Caches (sum of all):      
  L1d:                    32 KiB (1 instance)
  L1i:                    32 KiB (1 instance)
  L2:                     256 KiB (1 instance)
  L3:                     45 MiB (1 instance)
  NUMA:                     
  NUMA node(s):           1
  NUMA node0 CPU(s):      0
  Vulnerabilities:          
  Gather data sampling:   Not affected
  Itlb multihit:          KVM: Mitigation: VMX unsupported
  L1tf:                   Mitigation; PTE Inversion
  Mds:                    Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
  Meltdown:               Mitigation; PTI
  Mmio stale data:        Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
  Reg file data sampling: Not affected
  Retbleed:               Not affected
  Spec rstack overflow:   Not affected
  Spec store bypass:      Vulnerable
  Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:             Mitigation; Retpolines; STIBP disabled; RSB filling; PBRSB-eIBRS Not affected; BHI Retpoline
  Srbds:                  Not affected
  Tsx async abort:        Not affected
  ```

- `/etc/os-release` : Contains the information about the system OS
  ```
  $ cat /etc/os-release 
  PRETTY_NAME="Ubuntu 24.04.1 LTS"
  NAME="Ubuntu"
  VERSION_ID="24.04"
  VERSION="24.04.1 LTS (Noble Numbat)"
  VERSION_CODENAME=noble
  ID=ubuntu
  ID_LIKE=debian
  HOME_URL="https://www.ubuntu.com/"
  SUPPORT_URL="https://help.ubuntu.com/"
  BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
  PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
  UBUNTU_CODENAME=noble
  LOGO=ubuntu-logo
  ```

# Networking commands

- `nc` : netcat command is acts like a cat command over network, maily used for operations related to TCP, UDP and Unix domain sockets, port scanning, port listening, port redirection, open remote connections, r/w data over n/w, n/w debugging etc
  ```
  # port scanning
  $ nc -vz localhost 80
  nc  -zv localhost 80
  Connection to localhost (127.0.0.1) 80 port [tcp/http] succeeded!
  
  $ nc  -zv localhost 443
  nc: connect to localhost (127.0.0.1) port 443 (tcp) failed: Connection refused
  ```
- `netstat` : stands for network statistics. It allows users to display network-related information and diagnose various networking issues
  ```
  # shows both listening and non listening sockets
  $ netstat -a
  netstat -a
  Active Internet connections (servers and established)
  Proto Recv-Q Send-Q Local Address           Foreign Address         State      
  tcp        0      0 0.0.0.0:http            0.0.0.0:*               LISTEN     
  tcp        0      0 _localdnsstub:domain    0.0.0.0:*               LISTEN     
  tcp        0      0 0.0.0.0:8000            0.0.0.0:*               LISTEN     
  tcp        0      0 _localdnsproxy:domain   0.0.0.0:*               LISTEN     
  tcp        0      0 ip-172-31-17-0.ec:40668 169.254.169.254:http    TIME_WAIT  
  tcp        0      0 ip-172-31-17-0.ec:52472 169.254.169.254:http    TIME_WAIT  
  tcp        0      0 ip-172-31-17-0.ec:58466 ec2-54-165-17-230.:http TIME_WAIT  
  tcp6       0      0 [::]:http               [::]:*                  LISTEN     
  tcp6       0      0 [::]:ssh                [::]:*                  LISTEN     
  tcp6       0   1448 ip-172-31-17-0.ec2.:ssh 61.2.9.161:62185        ESTABLISHED
  udp        0      0 localhost:323           0.0.0.0:*                          
  udp        0      0 _localdnsproxy:domain   0.0.0.0:*                          
  udp        0      0 _localdnsstub:domain    0.0.0.0:*                          
  udp        0      0 ip-172-31-17-0.e:bootpc 0.0.0.0:*                          
  udp6       0      0 ip6-localhost:323       [::]:*                             
  raw6       0      0 [::]:ipv6-icmp          [::]:*                  7          
  Active UNIX domain sockets (servers and established)
  Proto RefCnt Flags       Type       State         I-Node   Path
  unix  3      [ ]         STREAM     CONNECTED     7544     
  unix  3      [ ]         STREAM     CONNECTED     8373     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     7543     
  unix  3      [ ]         SEQPACKET  CONNECTED     6975     
  unix  3      [ ]         DGRAM      CONNECTED     5807     
  unix  3      [ ]         STREAM     CONNECTED     5782     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     6536     /run/systemd/journal/stdout
  unix  3      [ ]         SEQPACKET  CONNECTED     6974     
  unix  3      [ ]         STREAM     CONNECTED     6609     
  unix  3      [ ]         STREAM     CONNECTED     6954     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     8646     
  unix  2      [ ]         DGRAM                    8422     /run/user/1000/systemd/notify
  unix  2      [ ACC ]     STREAM     LISTENING     8425     /run/user/1000/systemd/private
  unix  2      [ ]         DGRAM      CONNECTED     7074     
  unix  2      [ ACC ]     STREAM     LISTENING     8434     /run/user/1000/bus
  unix  2      [ ]         DGRAM      CONNECTED     2456     
  unix  3      [ ]         STREAM     CONNECTED     10116    /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     4152     /run/systemd/journal/stdout
  unix  2      [ ACC ]     STREAM     LISTENING     8438     /run/user/1000/gnupg/S.dirmngr
  unix  2      [ ACC ]     STREAM     LISTENING     8440     /run/user/1000/gnupg/S.gpg-agent.browser
  unix  2      [ ACC ]     STREAM     LISTENING     8442     /run/user/1000/gnupg/S.gpg-agent.extra
  unix  3      [ ]         STREAM     CONNECTED     5781     
  unix  2      [ ACC ]     STREAM     LISTENING     8455     /run/user/1000/gnupg/S.gpg-agent
  unix  2      [ ACC ]     STREAM     LISTENING     8459     /run/user/1000/gnupg/S.keyboxd
  unix  2      [ ACC ]     STREAM     LISTENING     8461     /run/user/1000/pk-debconf-socket
  unix  3      [ ]         STREAM     CONNECTED     6610     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     8645     
  unix  2      [ ACC ]     STREAM     LISTENING     8463     /run/user/1000/snapd-session-agent.socket
  unix  2      [ ACC ]     STREAM     LISTENING     8496     /run/user/1000/gnupg/S.gpg-agent.ssh
  unix  2      [ ]         DGRAM      CONNECTED     2035     
  unix  3      [ ]         STREAM     CONNECTED     7063     
  unix  2      [ ]         DGRAM      CONNECTED     8399     
  unix  3      [ ]         STREAM     CONNECTED     6277     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     6276     /run/dbus/system_bus_socket
  unix  2      [ ]         DGRAM      CONNECTED     8382     
  unix  2      [ ]         DGRAM                    6596     
  unix  3      [ ]         DGRAM      CONNECTED     1704     
  unix  3      [ ]         STREAM     CONNECTED     7534     
  unix  2      [ ]         STREAM     CONNECTED     8241     
  unix  2      [ ]         DGRAM      CONNECTED     6933     
  unix  3      [ ]         STREAM     CONNECTED     8427     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     10115    
  unix  3      [ ]         DGRAM      CONNECTED     1703     
  unix  2      [ ]         DGRAM      CONNECTED     5796     
  unix  3      [ ]         STREAM     CONNECTED     2446     
  unix  3      [ ]         STREAM     CONNECTED     6150     
  unix  2      [ ]         DGRAM      CONNECTED     8301     
  unix  3      [ ]         STREAM     CONNECTED     6775     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     6774     
  unix  3      [ ]         DGRAM      CONNECTED     1702     /run/systemd/notify
  unix  2      [ ACC ]     STREAM     LISTENING     4884     /run/systemd/resolve/io.systemd.Resolve
  unix  2      [ ACC ]     STREAM     LISTENING     1705     /run/systemd/private
  unix  3      [ ]         STREAM     CONNECTED     7064     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     6683     /run/systemd/journal/stdout
  unix  2      [ ACC ]     STREAM     LISTENING     4885     /run/systemd/resolve/io.systemd.Resolve.Monitor
  unix  2      [ ACC ]     STREAM     LISTENING     1707     /run/systemd/userdb/io.systemd.DynamicUser
  unix  3      [ ]         DGRAM      CONNECTED     8423     
  unix  2      [ ACC ]     STREAM     LISTENING     1708     /run/systemd/io.systemd.ManagedOOM
  unix  3      [ ]         STREAM     CONNECTED     6547     
  unix  3      [ ]         STREAM     CONNECTED     6548     /run/systemd/journal/stdout
  unix  2      [ ACC ]     STREAM     LISTENING     1723     /run/lvm/lvmpolld.socket
  unix  3      [ ]         STREAM     CONNECTED     8370     
  unix  2      [ ]         DGRAM                    1726     /run/systemd/journal/syslog
  unix  3      [ ]         DGRAM      CONNECTED     2460     
  unix  2      [ ACC ]     STREAM     LISTENING     1728     /run/systemd/fsck.progress
  unix  12     [ ]         DGRAM      CONNECTED     1730     /run/systemd/journal/dev-log
  unix  8      [ ]         DGRAM      CONNECTED     1732     /run/systemd/journal/socket
  unix  2      [ ACC ]     STREAM     LISTENING     1734     /run/systemd/journal/stdout
  unix  2      [ ACC ]     SEQPACKET  LISTENING     1739     /run/udev/control
  unix  3      [ ]         STREAM     CONNECTED     6951     
  unix  3      [ ]         STREAM     CONNECTED     2772     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     10097    
  unix  3      [ ]         STREAM     CONNECTED     6842     
  unix  3      [ ]         STREAM     CONNECTED     6682     
  unix  2      [ ]         DGRAM      CONNECTED     6103     
  unix  2      [ ACC ]     STREAM     LISTENING     1840     /run/systemd/journal/io.systemd.journal
  unix  3      [ ]         DGRAM      CONNECTED     2461     
  unix  3      [ ]         STREAM     CONNECTED     6843     
  unix  3      [ ]         STREAM     CONNECTED     6748     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     2389     
  unix  3      [ ]         STREAM     CONNECTED     6720     
  unix  3      [ ]         STREAM     CONNECTED     4151     
  unix  3      [ ]         STREAM     CONNECTED     8193     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     6534     
  unix  2      [ ]         DGRAM      CONNECTED     1845     
  unix  3      [ ]         DGRAM      CONNECTED     5810     
  unix  3      [ ]         STREAM     CONNECTED     6519     /run/dbus/system_bus_socket
  unix  2      [ ]         DGRAM      CONNECTED     6162     
  unix  3      [ ]         DGRAM      CONNECTED     5808     
  unix  3      [ ]         STREAM     CONNECTED     6152     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     2783     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     8192     
  unix  3      [ ]         DGRAM      CONNECTED     5809     
  unix  2      [ ]         DGRAM      CONNECTED     10099    
  unix  2      [ ]         DGRAM      CONNECTED     6727     
  unix  3      [ ]         STREAM     CONNECTED     10098    /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     6721     
  unix  3      [ ]         DGRAM      CONNECTED     8424     
  unix  2      [ ]         DGRAM      CONNECTED     4493     
  unix  2      [ ]         DGRAM      CONNECTED     7272     
  unix  3      [ ]         STREAM     CONNECTED     7428     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     7343     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     7245     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     7244     
  unix  2      [ ]         DGRAM      CONNECTED     7426     
  unix  3      [ ]         STREAM     CONNECTED     6430     /run/systemd/journal/stdout
  unix  2      [ ACC ]     STREAM     LISTENING     5991     /run/acpid.socket
  unix  2      [ ACC ]     STREAM     LISTENING     5996     /run/dbus/system_bus_socket
  unix  2      [ ACC ]     STREAM     LISTENING     6001     /run/lxd-installer.socket
  unix  3      [ ]         STREAM     CONNECTED     6307     /run/systemd/journal/stdout
  unix  2      [ ACC ]     STREAM     LISTENING     6003     /run/snapd.socket
  unix  2      [ ACC ]     STREAM     LISTENING     6004     /run/snapd-snap.socket
  unix  2      [ ACC ]     STREAM     LISTENING     6013     /run/uuidd/request
  unix  2      [ ACC ]     STREAM     LISTENING     8076     /var/lib/amazon/ssm/ipc/health
  unix  2      [ ]         DGRAM      CONNECTED     6272     
  unix  2      [ ACC ]     STREAM     LISTENING     8077     /var/lib/amazon/ssm/ipc/termination
  unix  3      [ ]         STREAM     CONNECTED     6306     
  unix  3      [ ]         STREAM     CONNECTED     6273     
  unix  3      [ ]         STREAM     CONNECTED     7887     
  unix  3      [ ]         STREAM     CONNECTED     7535     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     6390     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     6274     
  unix  3      [ ]         STREAM     CONNECTED     6428     
  unix  3      [ ]         STREAM     CONNECTED     7427     
  unix  3      [ ]         STREAM     CONNECTED     6189     /run/systemd/journal/stdout
  unix  2      [ ACC ]     STREAM     LISTENING     2973     /run/systemd/io.systemd.sysext
  unix  3      [ ]         STREAM     CONNECTED     6187     
  unix  3      [ ]         STREAM     CONNECTED     7888     /run/dbus/system_bus_socket
  unix  2      [ ]         DGRAM                    6973     /run/chrony/chronyd.sock
  unix  3      [ ]         STREAM     CONNECTED     6389     
  unix  3      [ ]         STREAM     CONNECTED     7341     
  unix  2      [ ACC ]     STREAM     LISTENING     1725     @/org/kernel/linux/storage/multipathd
  unix  3      [ ]         STREAM     CONNECTED     8426     @78d805b753f0a7ca/bus/systemd/bus-system
  unix  3      [ ]         STREAM     CONNECTED     6518     @291ff045cc5cf667/bus/systemd/bus-api-system
  unix  2      [ ACC ]     STREAM     LISTENING     5998     @ISCSIADM_ABSTRACT_NAMESPACE
  unix  3      [ ]         STREAM     CONNECTED     6747     @254c75110228d4f0/bus/systemd-logind/system
  unix  3      [ ]         STREAM     CONNECTED     6000     @ec1c5be8f825cf32/bus/systemd-network/bus-api-network
  unix  3      [ ]         STREAM     CONNECTED     5999     @60783b4b2b317577/bus/systemd-resolve/bus-api-resolve
  ```
  ```
  # displays all tcp ports
  $ netstat -at
  Active Internet connections (servers and established)
  Proto Recv-Q Send-Q Local Address           Foreign Address         State      
  tcp        0      0 0.0.0.0:http            0.0.0.0:*               LISTEN     
  tcp        0      0 _localdnsstub:domain    0.0.0.0:*               LISTEN     
  tcp        0      0 0.0.0.0:8000            0.0.0.0:*               LISTEN     
  tcp        0      0 _localdnsproxy:domain   0.0.0.0:*               LISTEN     
  tcp6       0      0 [::]:http               [::]:*                  LISTEN     
  tcp6       0      0 [::]:ssh                [::]:*                  LISTEN     
  tcp6       0   1364 ip-172-31-17-0.ec2.:ssh 61.2.9.161:62185        ESTABLISHED
  ```
  ```
  # display all udp ports
  $ netstat -au
  Active Internet connections (servers and established)
  Proto Recv-Q Send-Q Local Address           Foreign Address         State      
  udp        0      0 localhost:323           0.0.0.0:*                          
  udp        0      0 _localdnsproxy:domain   0.0.0.0:*                          
  udp        0      0 _localdnsstub:domain    0.0.0.0:*                          
  udp        0      0 ip-172-31-17-0.e:bootpc 0.0.0.0:*                          
  udp6       0      0 ip6-localhost:323       [::]:*
  ```
  ```
  # display only active listening connections
  $ netstat -l
  Active Internet connections (servers and established)
  Proto Recv-Q Send-Q Local Address           Foreign Address         State      
  tcp        0      0 0.0.0.0:http            0.0.0.0:*               LISTEN     
  tcp        0      0 _localdnsstub:domain    0.0.0.0:*               LISTEN     
  tcp        0      0 0.0.0.0:8000            0.0.0.0:*               LISTEN     
  tcp        0      0 _localdnsproxy:domain   0.0.0.0:*               LISTEN     
  tcp6       0      0 [::]:http               [::]:*                  LISTEN     
  tcp6       0      0 [::]:ssh                [::]:*                  LISTEN     
  tcp6       0   1212 ip-172-31-17-0.ec2.:ssh 61.2.9.161:62185        ESTABLISHED
  udp        0      0 localhost:323           0.0.0.0:*                          
  udp        0      0 _localdnsproxy:domain   0.0.0.0:*                          
  udp        0      0 _localdnsstub:domain    0.0.0.0:*                          
  udp        0      0 ip-172-31-17-0.e:bootpc 0.0.0.0:*                          
  udp        0      0 ip-172-31-17-0.ec:37040 alphyn.canonical.co:ntp ESTABLISHED
  udp6       0      0 ip6-localhost:323       [::]:*                             
  raw6       0      0 [::]:ipv6-icmp          [::]:*                  7          
  Active UNIX domain sockets (servers and established)
  Proto RefCnt Flags       Type       State         I-Node   Path
  unix  3      [ ]         STREAM     CONNECTED     13468    /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     7544     
  unix  3      [ ]         STREAM     CONNECTED     8373     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     7543     
  unix  3      [ ]         SEQPACKET  CONNECTED     6975     
  unix  3      [ ]         DGRAM      CONNECTED     5807     
  unix  3      [ ]         STREAM     CONNECTED     5782     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     6536     /run/systemd/journal/stdout
  unix  3      [ ]         SEQPACKET  CONNECTED     6974     
  unix  3      [ ]         STREAM     CONNECTED     6609     
  unix  3      [ ]         STREAM     CONNECTED     6954     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     8646     
  unix  2      [ ]         DGRAM                    8422     /run/user/1000/systemd/notify
  unix  2      [ ACC ]     STREAM     LISTENING     8425     /run/user/1000/systemd/private
  unix  2      [ ]         DGRAM      CONNECTED     7074     
  unix  2      [ ACC ]     STREAM     LISTENING     8434     /run/user/1000/bus
  unix  2      [ ]         DGRAM      CONNECTED     2456     
  unix  3      [ ]         STREAM     CONNECTED     10116    /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     4152     /run/systemd/journal/stdout
  unix  2      [ ACC ]     STREAM     LISTENING     8438     /run/user/1000/gnupg/S.dirmngr
  unix  2      [ ACC ]     STREAM     LISTENING     8440     /run/user/1000/gnupg/S.gpg-agent.browser
  unix  2      [ ACC ]     STREAM     LISTENING     8442     /run/user/1000/gnupg/S.gpg-agent.extra
  unix  3      [ ]         STREAM     CONNECTED     5781     
  unix  2      [ ACC ]     STREAM     LISTENING     8455     /run/user/1000/gnupg/S.gpg-agent
  unix  2      [ ACC ]     STREAM     LISTENING     8459     /run/user/1000/gnupg/S.keyboxd
  unix  2      [ ACC ]     STREAM     LISTENING     8461     /run/user/1000/pk-debconf-socket
  unix  3      [ ]         STREAM     CONNECTED     6610     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     8645     
  unix  2      [ ACC ]     STREAM     LISTENING     8463     /run/user/1000/snapd-session-agent.socket
  unix  2      [ ACC ]     STREAM     LISTENING     8496     /run/user/1000/gnupg/S.gpg-agent.ssh
  unix  2      [ ]         DGRAM      CONNECTED     2035     
  unix  3      [ ]         STREAM     CONNECTED     13445    
  unix  3      [ ]         STREAM     CONNECTED     7063     
  unix  2      [ ]         DGRAM      CONNECTED     8399     
  unix  3      [ ]         STREAM     CONNECTED     6277     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     6276     /run/dbus/system_bus_socket
  unix  2      [ ]         DGRAM      CONNECTED     8382     
  unix  2      [ ]         DGRAM                    6596     
  unix  3      [ ]         DGRAM      CONNECTED     1704     
  unix  3      [ ]         STREAM     CONNECTED     7534     
  unix  2      [ ]         STREAM     CONNECTED     8241     
  unix  2      [ ]         DGRAM      CONNECTED     6933     
  unix  3      [ ]         STREAM     CONNECTED     8427     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     10115    
  unix  3      [ ]         DGRAM      CONNECTED     1703     
  unix  2      [ ]         DGRAM      CONNECTED     5796     
  unix  3      [ ]         STREAM     CONNECTED     2446     
  unix  3      [ ]         STREAM     CONNECTED     6150     
  unix  2      [ ]         DGRAM      CONNECTED     8301     
  unix  3      [ ]         STREAM     CONNECTED     6775     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     6774     
  unix  3      [ ]         DGRAM      CONNECTED     1702     /run/systemd/notify
  unix  2      [ ACC ]     STREAM     LISTENING     4884     /run/systemd/resolve/io.systemd.Resolve
  unix  2      [ ACC ]     STREAM     LISTENING     1705     /run/systemd/private
  unix  3      [ ]         STREAM     CONNECTED     7064     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     6683     /run/systemd/journal/stdout
  unix  2      [ ACC ]     STREAM     LISTENING     4885     /run/systemd/resolve/io.systemd.Resolve.Monitor
  unix  2      [ ACC ]     STREAM     LISTENING     1707     /run/systemd/userdb/io.systemd.DynamicUser
  unix  3      [ ]         DGRAM      CONNECTED     8423     
  unix  2      [ ACC ]     STREAM     LISTENING     1708     /run/systemd/io.systemd.ManagedOOM
  unix  3      [ ]         STREAM     CONNECTED     6547     
  unix  3      [ ]         STREAM     CONNECTED     6548     /run/systemd/journal/stdout
  unix  2      [ ACC ]     STREAM     LISTENING     1723     /run/lvm/lvmpolld.socket
  unix  3      [ ]         STREAM     CONNECTED     8370     
  unix  2      [ ]         DGRAM                    1726     /run/systemd/journal/syslog
  unix  3      [ ]         DGRAM      CONNECTED     2460     
  unix  2      [ ACC ]     STREAM     LISTENING     1728     /run/systemd/fsck.progress
  unix  3      [ ]         STREAM     CONNECTED     13446    /run/systemd/journal/stdout
  unix  12     [ ]         DGRAM      CONNECTED     1730     /run/systemd/journal/dev-log
  unix  9      [ ]         DGRAM      CONNECTED     1732     /run/systemd/journal/socket
  unix  2      [ ACC ]     STREAM     LISTENING     1734     /run/systemd/journal/stdout
  unix  2      [ ACC ]     SEQPACKET  LISTENING     1739     /run/udev/control
  unix  3      [ ]         STREAM     CONNECTED     6951     
  unix  3      [ ]         STREAM     CONNECTED     2772     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     10097    
  unix  3      [ ]         STREAM     CONNECTED     6842     
  unix  3      [ ]         STREAM     CONNECTED     6682     
  unix  2      [ ]         DGRAM      CONNECTED     6103     
  unix  2      [ ACC ]     STREAM     LISTENING     1840     /run/systemd/journal/io.systemd.journal
  unix  3      [ ]         DGRAM      CONNECTED     2461     
  unix  3      [ ]         STREAM     CONNECTED     6843     
  unix  3      [ ]         STREAM     CONNECTED     6748     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     2389     
  unix  3      [ ]         STREAM     CONNECTED     6720     
  unix  3      [ ]         STREAM     CONNECTED     4151     
  unix  3      [ ]         STREAM     CONNECTED     8193     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     6534     
  unix  2      [ ]         DGRAM      CONNECTED     1845     
  unix  3      [ ]         DGRAM      CONNECTED     5810     
  unix  3      [ ]         STREAM     CONNECTED     6519     /run/dbus/system_bus_socket
  unix  2      [ ]         DGRAM      CONNECTED     6162     
  unix  3      [ ]         DGRAM      CONNECTED     5808     
  unix  3      [ ]         STREAM     CONNECTED     6152     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     2783     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     8192     
  unix  2      [ ]         DGRAM      CONNECTED     13462    
  unix  3      [ ]         DGRAM      CONNECTED     5809     
  unix  2      [ ]         DGRAM      CONNECTED     10099    
  unix  2      [ ]         DGRAM      CONNECTED     6727     
  unix  3      [ ]         STREAM     CONNECTED     10098    /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     6721     
  unix  3      [ ]         DGRAM      CONNECTED     8424     
  unix  2      [ ]         DGRAM      CONNECTED     4493     
  unix  2      [ ]         DGRAM      CONNECTED     7272     
  unix  3      [ ]         STREAM     CONNECTED     7428     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     7343     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     7245     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     7244     
  unix  2      [ ]         DGRAM      CONNECTED     7426     
  unix  3      [ ]         STREAM     CONNECTED     6430     /run/systemd/journal/stdout
  unix  2      [ ACC ]     STREAM     LISTENING     5991     /run/acpid.socket
  unix  2      [ ACC ]     STREAM     LISTENING     5996     /run/dbus/system_bus_socket
  unix  2      [ ACC ]     STREAM     LISTENING     6001     /run/lxd-installer.socket
  unix  3      [ ]         STREAM     CONNECTED     6307     /run/systemd/journal/stdout
  unix  2      [ ACC ]     STREAM     LISTENING     6003     /run/snapd.socket
  unix  2      [ ACC ]     STREAM     LISTENING     6004     /run/snapd-snap.socket
  unix  2      [ ACC ]     STREAM     LISTENING     6013     /run/uuidd/request
  unix  2      [ ACC ]     STREAM     LISTENING     8076     /var/lib/amazon/ssm/ipc/health
  unix  2      [ ]         DGRAM      CONNECTED     6272     
  unix  2      [ ACC ]     STREAM     LISTENING     8077     /var/lib/amazon/ssm/ipc/termination
  unix  3      [ ]         STREAM     CONNECTED     6306     
  unix  3      [ ]         STREAM     CONNECTED     6273     
  unix  3      [ ]         STREAM     CONNECTED     7887     
  unix  3      [ ]         STREAM     CONNECTED     7535     /run/dbus/system_bus_socket
  unix  3      [ ]         STREAM     CONNECTED     6390     /run/systemd/journal/stdout
  unix  3      [ ]         STREAM     CONNECTED     6274     
  unix  3      [ ]         STREAM     CONNECTED     6428     
  unix  3      [ ]         STREAM     CONNECTED     7427     
  unix  3      [ ]         STREAM     CONNECTED     6189     /run/systemd/journal/stdout
  unix  2      [ ACC ]     STREAM     LISTENING     2973     /run/systemd/io.systemd.sysext
  unix  3      [ ]         STREAM     CONNECTED     6187     
  unix  3      [ ]         STREAM     CONNECTED     7888     /run/dbus/system_bus_socket
  unix  2      [ ]         DGRAM                    6973     /run/chrony/chronyd.sock
  unix  3      [ ]         STREAM     CONNECTED     6389     
  unix  3      [ ]         STREAM     CONNECTED     7341     
  unix  2      [ ACC ]     STREAM     LISTENING     1725     @/org/kernel/linux/storage/multipathd
  unix  3      [ ]         STREAM     CONNECTED     13467    @68fa4b0a7a45fd8a/bus/systemd-timedat/system
  unix  3      [ ]         STREAM     CONNECTED     8426     @78d805b753f0a7ca/bus/systemd/bus-system
  unix  3      [ ]         STREAM     CONNECTED     6518     @291ff045cc5cf667/bus/systemd/bus-api-system
  unix  2      [ ACC ]     STREAM     LISTENING     5998     @ISCSIADM_ABSTRACT_NAMESPACE
  unix  3      [ ]         STREAM     CONNECTED     6747     @254c75110228d4f0/bus/systemd-logind/system
  unix  3      [ ]         STREAM     CONNECTED     6000     @ec1c5be8f825cf32/bus/systemd-network/bus-api-network
  unix  3      [ ]         STREAM     CONNECTED     5999     @60783b4b2b317577/bus/systemd-resolve/bus-api-resolve
  ```
  ```
  # displays network statistics
  $ netstat -s
  
  Ip:
  Forwarding: 2
  2106 total packets received
  4 with invalid addresses
  0 forwarded
  0 incoming packets discarded
  2102 incoming packets delivered
  1956 requests sent out
  OutTransmits: 1956
  Icmp:
  3 ICMP messages received
  0 input ICMP message failed
  ICMP input histogram:
  destination unreachable: 3
  3 ICMP messages sent
  0 ICMP messages failed
  ICMP output histogram:
  destination unreachable: 3
  IcmpMsg:
  InType3: 3
  OutType3: 3
  Tcp:
  95 active connection openings
  4 passive connection openings
  2 failed connection attempts
  2 connection resets received
  1 connections established
  1788 segments received
  1728 segments sent out
  17 segments retransmitted
  0 bad segments received
  70 resets sent
  Udp:
  308 packets received
  3 packets to unknown port received
  0 packet receive errors
  313 packets sent
  0 receive buffer errors
  0 send buffer errors
  UdpLite:
  TcpExt:
  2 resets received for embryonic SYN_RECV sockets
  3 TCP sockets finished time wait in fast timer
  13 delayed acks sent
  Quick ack mode was activated 1 times
  9 packet headers predicted
  529 acknowledgments not containing data payload received
  239 predicted acknowledgments
  TCPTimeouts: 15
  TCPLossProbes: 1
  TCPLossProbeRecovery: 1
  TCPDSACKOldSent: 1
  1 connections reset due to unexpected data
  1 connections reset due to early user close
  TCPRcvCoalesce: 103
  TCPOFOQueue: 2
  TCPAutoCorking: 52
  TCPSynRetrans: 16
  TCPOrigDataSent: 1159
  TCPHystartDelayDetect: 1
  TCPHystartDelayCwnd: 89
  TCPDelivered: 1237
  IpExt:
  InOctets: 788784
  OutOctets: 483581
  InNoECTPkts: 2341
  MPTcpExt:
  ```
  