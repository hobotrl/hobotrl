import time
import subprocess

if __name__ == '__main__':
        while True:
            cmd = 'netstat -an | wc -l'
            lines = subprocess.check_output(cmd.split(' '))
            print time.strftime("%H:%M:%S", time.gmtime(time.time())),
            print len(lines)
            time.sleep(5.0)
    #except Exception as e:
    #    print "End"
    #    print e.message
