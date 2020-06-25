import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import pygame, sys
from pygame.locals import *
import math
import numpy as np
from random import randint
INPUT = open("data\optimal.csv", "r")
pygame.init()
FPS = 30
FramePerSec = pygame.time.Clock()
CELL  = (247, 247, 247)
GRAY  = (127, 127, 127)
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DISPLAYSURF = pygame.display.set_mode((1050,770))
DISPLAYSURF.fill(WHITE)
pygame.display.set_caption("lang prediction")
wordInput = "---------------"
wordIndex = 0
mouse_x,mouse_y = 0,0
icon = pygame.image.load('icon.png')
pygame.display.set_icon(icon)
english,spanish,french,german,japanese,unknown = [],[],[],[],[],[]
clk,numIter,isTraining = 0,0,False
words = ["english", "spanish", "french", "german", "japanese", "unknown"]
button1Prev,button1Current,button2Prev,button2Current = 0,0,0,0

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(15*26, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def EncodeWord(s):
    result = torch.zeros(15, 26)
    one = torch.tensor([1])
    for i in range(15):
        if(s[i] != '-'):
            result[i][ord(s[i])-ord('A')] = one
    return result.view(-1, 15*26)

def GetPred(s):
    s = s.upper()
    X = EncodeWord(s)
    if(s == "---------------"):
        return torch.tensor([5])
    else:
        return torch.argmax(net(X.view(-1, 15*26))[0])

def GetConfidence(s):
    s = s.upper()
    X = EncodeWord(s)
    result = net(X.view(-1, 15*26))
    return result.detach().numpy()

def IsCorrect(s):
    final = True
    for i in range(len(s)):
        if(not (s[i] == '-' or (ord(s[i]) >= ord('A') and ord(s[i]) <= ord('s')))):
           final = False
           break
    return final

def doOneIteration(custom=False, word=None, label=0):
    if(custom == False):
        net.zero_grad()
        indexLang = randint(0, 5)
        lenght = 0
        inputLabel = words[indexLang].upper()
        inputWord = ""
        if(indexLang == 0): inputWord = english[randint(0, len(english)-1)]; 
        elif(indexLang == 1): inputWord = spanish[randint(0, len(spanish)-1)]; 
        elif(indexLang == 2): inputWord = french[randint(0, len(french)-1)];
        elif(indexLang == 3): inputWord = german[randint(0, len(german)-1)];
        elif(indexLang == 4): inputWord = japanese[randint(0, len(japanese)-1)];
        elif(indexLang == 5): inputWord = unknown[randint(0, len(unknown)-1)]; 

        endRange = 15-len(inputWord)
        for i in range(endRange):
            inputWord += '-'
        if(IsCorrect(inputWord) == True):
            predOutput = net(EncodeWord(inputWord).view(-1, 15*26))
            actualOutput = torch.tensor([indexLang])
            loss = F.nll_loss(predOutput, actualOutput)
            loss.backward()
            optimizer.step()
    else:
        predOutput = net(EncodeWord(word).view(-1, 15*26))
        actualOutput = torch.tensor([label])
        loss = F.nll_loss(predOutput, actualOutput)
        loss.backward()
        optimizer.step()
        
def change_char(s, p, r):
    return s[:p]+r+s[p+1:]

def getExpectedOutput(s):
    res = "UNKNOWN"
    if(s in english): res = "ENGLISH";
    elif(s in spanish): res = "SPANISH";
    elif(s in french): res = "FRENCH";
    elif(s in german): res = "GERMAN";
    elif(s in japanese): res = "JAPANESE";
    if(s == ""): res = "UNKNOWN";
    return res

def DrawScene():
    global button1Prev,button1Current,button2Prev,button2Current
    confiences = GetConfidence(wordInput)
    OutputConf = -math.inf
    for i in range(len(confiences[0])):
        if(OutputConf < confiences[0][i]):
            OutputConf = confiences[0][i]
    OutputConf = round(OutputConf, 1)
    DISPLAYSURF.fill(WHITE) 
    font = pygame.font.Font("freesansbold.ttf", 32)
    text = font.render('Input word:', True, BLACK, WHITE)
    textRect = text.get_rect()
    textRect.center = (95, 80)
    DISPLAYSURF.blit(text, textRect)

    real = ""
    for i in range(15):
        if(wordInput[i] != '-'):
            real += wordInput[i]
    lenght = len(real)
    font = pygame.font.Font("freesansbold.ttf", 28)
    text = font.render(real.upper(), True, BLUE, WHITE)
    textRect = text.get_rect()
    textRect.center = (300//2, 120)
    DISPLAYSURF.blit(text, textRect)

    font = pygame.font.Font("freesansbold.ttf", 32)
    text = font.render("1 - toggle training", True, BLACK, WHITE)
    textRect = text.get_rect()
    textRect.center = (300//2, 400)
    DISPLAYSURF.blit(text, textRect)

    text = font.render("2 - one training", True, BLACK, WHITE)
    textRect = text.get_rect()
    textRect.center = (300//2-20, 440)
    DISPLAYSURF.blit(text, textRect)

    value = GetPred(wordInput)
    prediction = (words[value]).upper()
    expected = getExpectedOutput(real.upper())
    text = font.render("Confidence: " + str(OutputConf) + "%", True, BLACK, WHITE)
    textRect = text.get_rect()
    textRect.center = (300//2, 360)
    DISPLAYSURF.blit(text, textRect)

    text = font.render(prediction, True, BLUE, WHITE)
    textRect = text.get_rect()
    textRect.center = (300//2, 280)
    DISPLAYSURF.blit(text, textRect)

    text = font.render("Prediction:", True, BLACK, WHITE)
    textRect = text.get_rect()
    textRect.center = (300//2-60, 240)
    DISPLAYSURF.blit(text, textRect)
    text = font.render(expected, True, BLUE, WHITE)
    
    textRect = text.get_rect()
    textRect.center = (300//2, 200)
    DISPLAYSURF.blit(text, textRect)

    text = font.render("Expected Output:", True, BLACK, WHITE)
    textRect = text.get_rect()
    textRect.center = (300//2-5, 160)
    DISPLAYSURF.blit(text, textRect)

    text = font.render("Iteration #" + str(numIter), True, BLACK, WHITE)
    textRect = text.get_rect()
    textRect.center = (300//2, 30)
    DISPLAYSURF.blit(text, textRect)

    button1Color = GRAY
    button2Color = GRAY
    if(mouse_x >= 11 and mouse_x <= 11+276 and mouse_y >= 520 and mouse_y <= 520+70):
        button1Color = (170, 170, 170)
    if(mouse_x >= 11 and mouse_x <= 11+276 and mouse_y >= 610 and mouse_y <= 610+70):
        button2Color = (170, 170, 170)
    pygame.draw.rect(DISPLAYSURF,GRAY,(300,0,900,900))
    pygame.draw.rect(DISPLAYSURF,button1Color,(11,520,276,70))
    pygame.draw.rect(DISPLAYSURF,button2Color,(11,610,276,70))

    text = font.render("Save model", True, BLACK, button1Color)
    textRect = text.get_rect()
    textRect.center = (150, 555)
    DISPLAYSURF.blit(text, textRect)

    text = font.render("Load model", True, BLACK, button2Color)
    textRect = text.get_rect()
    textRect.center = (150, 645)
    DISPLAYSURF.blit(text, textRect)
    DrawNeuralNetwork()

    x_offset = 265
    for i in range(6):
        pygame.draw.circle(DISPLAYSURF, CELL, (1000, x_offset), 20)
        font = pygame.font.Font("freesansbold.ttf", 20)
        text = font.render(str(round(confiences[0][i], 1)), True, BLACK, CELL)
        textRect = text.get_rect()
        textRect.center = (1000, x_offset)
        DISPLAYSURF.blit(text, textRect)
        x_offset += 50

def map(x,in_min,in_max,out_min,out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def GetValuesLayer(n):
    mini = 1000000
    maxi = -1000000
    weights = None
    if(n == 1): weights = net.fc1.weight.data.numpy();
    elif(n == 2): weights = net.fc2.weight.data.numpy();
    elif(n == 3): weights = net.fc3.weight.data.numpy();
    elif(n == 4): weights = net.fc4.weight.data.numpy();

    for i in range(len(weights)):
        if(mini > weights[0][i]):
            mini = weights[0][i]
        if(maxi < weights[0][i]):
            maxi = weights[0][i]
    
    return [mini, maxi]

def absolut(n):
    if(n > 255):
        return 255
    elif(n < 0):
        return 0
    else:
        return n

def DrawNeuralNetwork():
    valLayer1 = GetValuesLayer(1)
    valLayer2 = GetValuesLayer(2)
    minLayer1,maxLayer1 = valLayer1[0],valLayer1[1]
    minLayer2,maxLayer2 = valLayer2[0],valLayer2[1]
    x_offset = 30
    for i in range(15):
        pygame.draw.circle(DISPLAYSURF, CELL, (340, x_offset), 20)
        font = pygame.font.Font("freesansbold.ttf", 32)
        text = font.render(wordInput[i].upper(), True, BLACK, CELL)
        textRect = text.get_rect()
        textRect.center = (340, x_offset)
        destinationOffset = 30
        weights = net.fc1.weight.data.numpy();
        for k in range(15):
            cellOffset = 30
            for j in range(15):
                COLOR = map(weights[k][j], minLayer1, maxLayer1, 0, 255)
                COLOR = int(absolut(COLOR))
                COLOR = [COLOR, COLOR, COLOR]
                pygame.draw.line(DISPLAYSURF, tuple(COLOR), (360, destinationOffset), (670-15, cellOffset), 3)
                cellOffset += 50
            destinationOffset += 50
        DISPLAYSURF.blit(text, textRect)
        x_offset += 50

    x_offset = 30
    for i in range(15):
        pygame.draw.circle(DISPLAYSURF, CELL, (670, x_offset), 20)
        destinationOffset = 30
        weights = net.fc2.weight.data.numpy()
        for k in range(15):
            cellOffset = 265
            for j in range(6):
                COLOR = map(weights[k][j], minLayer1, maxLayer1, 0, 255)
                COLOR = int(absolut(COLOR))
                COLOR = [COLOR, COLOR, COLOR]
                pygame.draw.line(DISPLAYSURF, tuple(COLOR), (690, destinationOffset), (1000-15, cellOffset), 3)
                cellOffset += 50
            destinationOffset += 50
        x_offset += 50

    x_offset = 265
    for i in range(6):
        pygame.draw.circle(DISPLAYSURF, CELL, (1000, x_offset), 20)
        x_offset += 50

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
        nr = getNumber(row)
        row = filterWord(row)
        if(nr == 1): english.append(row);
        elif(nr == 2): spanish.append(row);
        elif(nr == 3): french.append(row);
        elif(nr == 4): german.append(row);
        elif(nr == 5): japanese.append(row);
        elif(nr == 6): unknown.append(row);
        row = ""
INPUT.close()
    
while True:
    DrawScene()
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif(event.type == pygame.KEYDOWN):
            if(event.key >= pygame.K_a and event.key <= pygame.K_z and not isTraining):
                if(wordIndex < 15):
                    wordInput = change_char(wordInput, wordIndex, chr(event.key))
                    wordIndex += 1
            elif(event.key == K_BACKSPACE and not isTraining):
                if(wordIndex > 0):
                    wordIndex -= 1
                    wordInput = change_char(wordInput, wordIndex, "-")

            if(event.key == 49):
                numIter += 1
                for k in range(10):
                    doOneIteration()
            elif(event.key == 50):
                isTraining = not isTraining
    if pygame.mouse.get_pressed()[0]:
        if(mouse_x >= 11 and mouse_x <= 11+276 and mouse_y >= 520 and mouse_y <= 520+70):
            button1Current = 1
            if(button1Current == 1 and button1Prev == 0):
                torch.save(net.state_dict(), "model.pth")
        else:
            button1Current = 0
        if(mouse_x >= 11 and mouse_x <= 11+276 and mouse_y >= 610 and mouse_y <= 610+70):
            button2Current = 1
            if(button2Current == 1 and button2Prev == 0):
                net.load_state_dict(torch.load('model.pth'))
                numIter = 0
        else:
            button2Current = 0
    else:
        button1Current,button2Current = 0,0
        
    mouse_x, mouse_y = pygame.mouse.get_pos()[0],pygame.mouse.get_pos()[1]
    FramePerSec.tick(FPS)
    clk += 1
    
    if(clk%5 == 0 and isTraining):
        numIter += 1
        for k in range(10):
            doOneIteration()
    button1Prev = button1Current
    button2Prev = button2Current
