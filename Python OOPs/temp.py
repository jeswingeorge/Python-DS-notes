class Customer:
    def __init__(self, cust_id, cust_name, location):
        self.cust_id = cust_id
        self.cust_name = cust_name
        self.location = location
        
    def __str__(self):
        return(f"customer id: {self.cust_id}, customer name: {self.cust_name}, location: {self.location}")

list_of_customers = [Customer(101, 'Mark', 'US'),
Customer(102, 'Jane', 'Japan'),
Customer(103, 'Kumar', 'India')]

dict_of_customer={}

for obj in list_of_customers:
    dict_of_customer[obj.location] = obj

# print(dict_of_customer.keys())

# print(dict_of_customer['US'])

for key in dict_of_customer.keys():
    print(key,' : ',dict_of_customer[key])
