from run_learning import run_learning, run_demo

program = input('请选择需要执行的任务：\n 1.学习 \n 2.演示 \n')
if int(program) == 1:
    run_learning()
elif int(program) == 2:
    run_demo()
else:
    print('输入错误，程序结束')