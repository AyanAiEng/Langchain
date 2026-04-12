from pydantic import BaseModel

class Student(BaseModel):
    name:str = "ayan"

new_student = {"name":"AYnaali"}

""" 
in the pydantic dictionary the advantage is that it can throw error when not give str at the place of 
and it also to do other thins that typedict doesnot allow
 """
student = Student(** new_student)

print(student.name)
