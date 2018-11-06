# Ozeas - ozeasx@gmail.com
import subprocess


class Shell(object):
    # Run command in Linux shell and return result as a string
    @staticmethod
    def run(cmd):
        # print cmd
        return str(subprocess.check_output([cmd], shell=True))

    # Call script in background
    @staticmethod
    def call(script, args):
        subprocess.call(['./test.sh'])
