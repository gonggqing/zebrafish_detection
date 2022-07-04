from fishutil import *

class Zebrafish():

    def __init__(self, info):
        self.info = info

    def islarva(self):
        return self.info['dead'] == [] and self.info['unhatched embryo'] == []

    def isdead(self):
        return self.info['dead'] != []

    def isembryo(self):
        return self.info['unhatched embryo'] != []

    def isbent(self):
        return self.info['bent spine'] != []

    def isjawmal(self):
        return self.info['jaw malformation'] != []

    def ispedema(self):
        return self.info['pericardial edema'] != []

    def isyedema(self):
        return self.info['yolk edema'] != []

    def isheadamage(self):
        return self.info['head hemorrhage'] != []

    def isabsence(self):
        return self.info['swim bladder absence'] != []

    def haseye(self):
        return self.info['eye'] != []

    def hastail(self):
        return self.info['tail'] != []

    def haspine(self):
        return self.info['spine'] != [] or self.info['bent spine'] != []

    def getlength(self):
        if self.islarva() and self.haseye():
            return get_body(self.info)
        else:
            return 0

    def getcurve(self):
        if self.islarva():
            return get_curve(self.info)
        else:
            return 0

    def gethead(self):
        if self.islarva():
            if self.info['head'] != []:
                return self.info['head'][0]
            elif self.info['head hemorrhage'] != []:
                return self.info['head hemorrhage'][0]
            else:
                return 0
        else:
            return 0

    def geteye(self):
        if self.haseye():
            return self.info['eye'][0]
        else:
            return 0

    def getheart(self):
        if self.islarva():
            if self.info['heart'] != []:
                return self.info['heart'][0]
            elif self.info['pericardial edema'] != []:
                return self.info['pericardial edema'][0]
            else:
                return 0
        else:
            return 0

    def getyolk(self):
        if self.islarva():
            if self.info['yolk'] != []:
                return self.info['yolk'][0]
            elif self.info['yolk edema'] != []:
                return self.info['yolk edema'][0]
            else:
                return 0
        else:
            return 0

    def getail(self):
        if self.hastail():
            return get_tail_length(self.info)
        else:
            return 0

    def getspine(self):
        if self.haspine():
            return get_spine_length(self.info)
        else:
            return 0

    def getbladder(self):
        if self.islarva():
            if self.info['swim bladder'] != []:
                return self.info['swim bladder'][0]
            elif self.info['swim bladder absence'] != []:
                return self.info['swim bladder absence'][0]
            else:
                return 0
        else:
            return 0

    def getjaw(self):
        if self.islarva():
            if self.info['lower jaw'] != []:
                return self.info['lower jaw'][0]
            elif self.info['jaw malformation'] != []:
                return self.info['jaw malformation'][0]
            else:
                return 0
        else:
            return 0