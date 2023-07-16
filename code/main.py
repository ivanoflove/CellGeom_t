import subprocess

process1 = subprocess.Popen(['python3', 'main1.py'])
process2 = subprocess.Popen(['python3', 'main2.py'])
process3 = subprocess.Popen(['python3', 'main3.py'])
process4 = subprocess.Popen(['python3', 'main4.py'])
process5 = subprocess.Popen(['python3', 'main5.py'])
# process6 = subprocess.Popen(['python3', 'main6.py'])
# process7 = subprocess.Popen(['python3', 'main7.py'])
# process8 = subprocess.Popen(['python3', 'main8.py'])
# process9 = subprocess.Popen(['python3', 'main9.py'])
# process10 = subprocess.Popen(['python3', 'main10.py'])


process1.wait()
process2.wait()
process3.wait()
process4.wait()
process5.wait()
# process6.wait()
# process7.wait()
# process8.wait()
# process9.wait()
# process10.wait()
