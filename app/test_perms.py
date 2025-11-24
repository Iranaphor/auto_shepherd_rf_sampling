import os
import pathlib
import pwd
import grp

print("=== Process identity inside container ===")
print("UID:", os.getuid())
print("GID:", os.getgid())

try:
    print("User name:", pwd.getpwuid(os.getuid()).pw_name)
    print("Group name:", grp.getgrgid(os.getgid()).gr_name)
except KeyError:
    print("User/group name not found in /etc/passwd or /etc/group")

print("\n=== Writing test file to /data ===")
path = pathlib.Path("/data/permissions_test.txt")
path.write_text("hello from inside the container\n")

st = path.stat()
print("File owner UID:", st.st_uid)
print("File owner GID:", st.st_gid)

try:
    owner_user = pwd.getpwuid(st.st_uid).pw_name
except KeyError:
    owner_user = f"(no name for UID {st.st_uid})"

try:
    owner_group = grp.getgrgid(st.st_gid).gr_name
except KeyError:
    owner_group = f"(no name for GID {st.st_gid})"

print("File owner user:", owner_user)
print("File owner group:", owner_group)

print("\nDone. Now check this file on the host with: ls -l data/permissions_test.txt")
