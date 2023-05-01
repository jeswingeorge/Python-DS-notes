class Student:
    def __init__(self,student_id,age,marks) -> None:
        self.__student_id = student_id
        self.__age = age
        self.__marks = marks

    ## Setter methods
    def set_student_id(self, student_id):
        self.__student_id = student_id

    def set_age(self, age):
        self.__age = age

    def set_marks(self, marks):
        self.__marks = marks

    ## Getter methods
    def get_student_id(self):
        return(self.__student_id)
    
    def get_age(self):
        return(self.__age)
    
    def get_marks(self):
        return(self.__marks)
    
    ## Other methods

    def validate_marks(self):
        if (0<= self.get_marks() <= 100):
            return(True)
        else:
            return(False)
        
    def validate_age(self):
        if self.get_age() > 20:
            return(True)
        else:
            return(False)
     
    def check_qualification(self):
        if self.validate_age() and self.validate_marks():
            if self.get_marks()>=65:
                print("Student is eligible")
                return(True)
            else:
                return(False)
        print("Student is not eligible")
        return(False)

        
jeswin = Student('123', 31, 76)
jeswin.check_qualification()
    