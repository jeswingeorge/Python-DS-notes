class Ticket:
    counter = 0
    def __init__(self,passenger_name,source,destination) -> None:
        self.__passenger_name = passenger_name
        self.__source = source
        self.__destination = destination
        self.__ticket_id = self.generate_ticket()

    def __str__(self) -> str:
        return(f"Ticket_id: {self.get_ticket_id()}, Passenger name: {self.get_passenger_name()}, Source of journey: {self.get_source()}, Destination: {self.get_destination()}")
    
    def get_ticket_id(self):
        return self.__ticket_id
    
    def get_passenger_name(self):
        return(self.__passenger_name)
    
    def get_source(self):
        return self.__source
    
    def get_destination(self):
        return self.__destination
    
    def validate_source_destination(self):
        if self.get_source().lower()=="delhi":
            if self.get_destination().lower()=='mumbai' or self.get_destination().lower()=='chennai' or self.get_destination().lower()=='pune' or self.get_destination().lower()=='kolkata':
                return True
            
        else:
            print("Source must be Delhi")
            return False
    
    def generate_ticket(self):
        if self.validate_source_destination():
            Ticket.counter += 1
            prefix = self.get_source().upper()[0] + self.get_destination().upper()[0]
            if len(str(Ticket.counter))==1:
                self.__ticket_id = prefix + '0' +  str(Ticket.counter)
            else:
                self.__ticket_id = prefix +  str(Ticket.counter)

            return self.__ticket_id

        else:
            self.__ticket_id = None
            return self.__ticket_id



    

t1 = Ticket("jvg",'Kerala','Pune')
print(t1)


    
