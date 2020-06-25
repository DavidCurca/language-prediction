INPUT = open("not filterd.csv", "r")
OUTPUT = open("filterd.csv", "w")

def IsCorrect(s):
    if(" " in s or len(s) > 15):
        return False
    else:
        return True

def filterWord(s):
    res = ""
    for i in range(len(s)-2):
        res += s[i]
    return res

def getNumber(s):
    return int(s[len(s)-1])

row = ""
while True:
    c = INPUT.read(1)
    if not c:
        break
    if(c != '\n'):
        row += c

    if(c == '\n'):
        if(IsCorrect(row)):
            writerow = ""
            writerow += filterWord(row).upper()
            writerow += "," + str(getNumber(row))
            writerow += "\n"
            OUTPUT.write(writerow)
        row = ""

INPUT.close()
OUTPUT.close()
