import  Parameter
class farmer:
    def __init__(self, id, area,crop):
        self.id = id
        self.area = area
        #self.current_crop = 0
        self.current_crop = crop
        self.plant_month = 0
        self.crop_shedule = []
        self.supply_in_last_12 = [[] for i in range(Parameter.num_crop+1)]
        self.LastYearSupply = [[] for i in range(Parameter.num_crop+1)]
        self.profit_in_last_12 = []
    def FarmerShedule(self, num_month, action):
        current_action = { Parameter.Month_List[num_month]: action }
        self.crop_shedule.append(current_action)
    def GetActionShedule(self,num_month):
        if(len(self.crop_shedule) != 0):
            return self.crop_shedule[-1][Parameter.Month_List[num_month]]

