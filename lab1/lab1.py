class Student:
    def __init__(self, student_id, name, gender, room_number, phone_number):
        self.student_id = student_id
        self.name = name
        self.gender = gender
        self.room_number = room_number
        self.phone_number = phone_number

    def display_info(self):
        print(f"学号: {self.student_id}, 姓名: {self.name}, 性别: {self.gender}, 宿舍房间号: {self.room_number}, 联系号码: {self.phone_number}")


# 假定初始学生数据
students = {
    "10241111001":Student("10241111001", "张三", "男", "A101", "12345678901"),
    "10241111005":Student("10241111005", "李四", "男", "B202", "10987654321"),
    "10232220003":Student("10232220003", "王芳", "女", "C303", "11223344556"),
}

print(f"欢迎使用学生宿舍管理程序，目前程序内有{len(students)}名学生的信息。")
choice = input("请选择您要进行的操作：\n[1]根据学号查找学生信息\n[2]录入新的学生信息\n[3]显示所有学生信息\n[4]退出程序\n请输入选项（1/2/3/4）：")


while choice != '4':
    # 输入验证
    loop = True
    while loop:
        try:
            choice = int(choice)
            if choice in [1, 2, 3, 4]:
                loop = False
            else:
                choice = input("输入无效，请输入数字（1/2/3/4）：")
        except ValueError:
            choice = input("输入无效，请输入数字（1/2/3/4）：")

    ## 根据用户选择执行相应操作
    # 1查找学生信息
    q_early = False # 标记是否提前退出查询的变量
    if choice == 1:
        print('------------------------------------------------')
        student_id = input("请输入您要查找的学生学号：")
        found = False
        while not found:
            student_info = students.get(student_id)
            if student_info:
                print("找到该学生的信息：")
                student_info.display_info()
                found = True
            if not found:
                print("未找到该学生的信息。")
                student_id = input("请重新输入学生学号（或输入q退出学生信息查询）：")
                if student_id == 'q':
                    q_early = True # 标记提前退出查询
                    break

    # 2录入新学生信息
    elif choice == 2:
        print('------------------------------------------------')

        #对学生学号输入的检查
        student_id = input("请输入学生学号：")
        while not student_id:
            student_id = input("学号不能为空，请重新输入学生学号：")
        # 处理学号重复的情况
        number_unique = False
        while not number_unique:
            student_info = students.get(student_id)
            if student_info!=None:# 学号有重复
                number_unique = False
                print("该学号已存在，请选择覆盖已有信息或重新输入学号。\n")
                choice2 = input("[1]覆盖已有信息\n[2]重新输入学号\n请输入选项（1/2）：")
                if choice2 == '1':
                    del students[student_id]
                    number_unique = True
                elif choice2 == '2':
                    student_id = input("请输入学生学号：")
                    while not student_id:
                        student_id = input("学号不能为空，请重新输入学生学号：")
                else:
                    print("输入无效，请重新选择。")
            else:
                number_unique = True

        # 对学生姓名输入的检查
        name = input("请输入学生姓名：")
        while not name:
            name = input("姓名不能为空，请重新输入学生姓名：")

        # 对性别输入的检查
        gender = input("请输入学生性别（男/女）：")
        while gender not in ["男", "女"]:
            gender = input("性别输入无效，请输入“男”或“女”：")

        # 宿舍房间号输入
        room_number = input("请输入宿舍房间号：")
        while not room_number:
            room_number = input("宿舍房间号不能为空，请重新输入宿舍房间号：")

        # 对有效电话号码的检查
        phone_number = input("请输入联系号码：")
        while len(phone_number) != 11 or not phone_number.isdigit():
            phone_number = input("电话号码应为11位数字，请重新输入联系号码：")

        new_student_info = Student(student_id, name, gender, room_number, phone_number)
        students[student_id] = new_student_info
        print("新学生信息录入成功。")
    
    # 3显示所有学生信息
    elif choice == 3:
        print('------------------------------------------------')
        print(f"当前共有{len(students)}名学生的信息如下：")
        for student in students.values():
            student.display_info()
    
    # 处理提前退出查询的情况
    if q_early:
        q_early = False
        print('------------------------------------------------')
        choice = input(f'您已成功退出，请选择您接下来的操作：\n[1]根据学号查找学生信息\n[2]录入新的学生信息\n[3]显示所有学生信息\n[4]退出程序\n请输入选项（1/2/3/4）：')
        continue
    
    # 询问用户是否继续当前操作
    print('------------------------------------------------')
    conti = input("本次操作完成。是否继续重复当前操作？（y/n）：")
    is_y_n = False
    while not is_y_n:
        if conti.lower() == 'n':
            is_y_n = True
            print('------------------------------------------------')
            choice = input(f'请选择您接下来的操作：\n[1]根据学号查找学生信息\n[2]录入新的学生信息\n[3]显示所有学生信息\n[4]退出程序\n请输入选项（1/2/3/4）：')
        elif conti.lower() == 'y':
            is_y_n = True
            choice = str(choice)
        else:
            conti = input("输入无效。请重新选择是否继续当前操作？（y/n）：")

# 程序结束
print('------------------------------------------------')
print("感谢使用学生宿舍管理程序，再见！")
