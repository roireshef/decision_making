import numpy as np
T_dir=r'W:\SPAV\decision_making\test\proving_ground'
T_name="Floating_18_to_0.txt"
EW_EW_Straight_Az=1.670695822

Orig_traj=np.loadtxt(T_dir+'\\'+T_name, delimiter=",")

Orig_x=Orig_traj[0,0] # First x, y of map
Orig_y=Orig_traj[0,1]
Shift_X=Orig_traj[:,0]-Orig_x # shift all x and y to first x,y. map starts from 0,0
Shift_Y=Orig_traj[:,1]-Orig_y

Trans_X=Shift_X*np.cos(EW_EW_Straight_Az)+Shift_Y*np.sin(EW_EW_Straight_Az)
Trans_Y=-Shift_X*np.sin(EW_EW_Straight_Az)+Shift_Y*np.cos(EW_EW_Straight_Az)
Trans_AZ=Orig_traj[:,2]-EW_EW_Straight_Az


Transformed_traj=np.array([Trans_X, Trans_Y, Trans_AZ, Orig_traj[:,3], Orig_traj[:,4], Orig_traj[:,5]])
Transformed_traj=Transformed_traj.transpose()

np.savetxt('%s_Transformed.txt' %
             T_name[0:-4], Transformed_traj, delimiter=', ', newline='\n', fmt='%1.6f')
