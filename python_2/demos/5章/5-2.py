# class Student:
#     def __init__(self, name, age):
#         print("init...")
#         self.name = name
#         self.age = age
#
#     def __new__(cls, *args, **kwargs):
#         print("new...")
#         return super().__new__(cls)
#
#
# stu = Student("xm", 18)
# print(stu.name)


class Student:
    def __init__(self, name, age):
        print("init...")
        print(id(self))

        self.name = name
        self.age = age
        print(self.__dict__)

    def __new__(cls, *args, **kwargs):
        print("new...")
        obj = object.__new__(cls)
        print(id(obj))

        return obj


stu = Student("xm", 18)
