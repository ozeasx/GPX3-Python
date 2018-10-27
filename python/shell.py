# Ozeas - ozeasx@gmail.com
import subprocess

class Shell(object):
    # Run command in Linux shell and return result as a string
    def run(self, cmd):
        #print cmd
        return str(subprocess.check_output(cmd, shell = True))
