datapath = '/home/fabian/data/bacterial_growth_project/08_19_21/run_1'

deltatime=200000
timeincr=10000
for time in [12000000,13500000,13000000,14000000,14500000,14850000]:
    infile= '%s/grid-%d.pos'%(datapath,time)
    data=np.loadtxt(infile)
    xg=data[:,0]*0.0
    yg=data[:,1]*0.0
    rho=data[:,2]*0.0
    Q11=data[:,3]*0.0
    Q12=data[:,4]*0.0
    vx=data[:,5]*0.0
    vy=data[:,6]*0.0
    press=data[:,7]*0.0
    
    N=0
    
    for l_time in range(int(time-deltatime/2),int(time+deltatime/2),timeincr):
        infile= '%s/grid-%d.pos'%(datapath,l_time)
        data=np.loadtxt(infile)
        xg+=data[:,0]
        yg+=data[:,1]
        rho+=data[:,2]
        Q11+=data[:,3]
        Q12+=data[:,4]
        vx+=data[:,5]
        vy+=data[:,6]
        press+=data[:,7]
        N+=1
    xg=xg/N
    yg=yg/N
    rho=rho/N
    Q11=Q11/N
    Q12=Q12/N
    vx=vx/N
    vy=vy/N
    
    com_x= np.mean(xg)
    com_y= np.mean(yg)
    delx=1.0
    rad_dist=np.arange(2.0*delx,max(xg[np.where(rho>0)])-1.0,delx)
    Press_R=np.zeros(np.size(rad_dist))
    norm=np.zeros(np.size(rad_dist))
    V_r=np.zeros(np.size(rad_dist))
    V_phi=np.zeros(np.size(rad_dist))
    for i in range(0,np.size(xg)):
        l_x=com_x-xg[i]
        l_y=com_y-yg[i]
        dist= np.sqrt( (l_x)**2 + (l_y)**2 )
        angle= np.arctan2(l_y,l_x)+np.pi
        ind= np.where( ( (rad_dist-delx/2) <dist) & ( (rad_dist+delx/2) > dist)  )
        if(np.size(ind[0])> 0):
            Press_R[ind[0][0]]= Press_R[ind[0][0]]+ press[i]
            absv=np.sqrt(vx[i]**2+vy[i]**2)
            V_r[ind[0][0]]+=(vx[i]*np.cos(angle)+vy[i]*np.sin(angle))
            V_phi[ind[0][0]]+=(-vx[i]*np.sin(angle)+vy[i]*np.cos(angle))
            norm[ind[0][0]]=norm[ind[0][0]]+1
    Press_R=Press_R/(norm)
    V_r=V_r/norm
    V_phi=V_phi/norm
    #print(np.mean(V_r))
    plt.plot(rad_dist,Press_R,"o")
    #plt.plot(rad_dist,V_r/rad_dist,"o", label=time*0.01)#, label="$v_r/|\mathbf{v}|$")
    #plt.plot(rad_dist,V_phi,"x")#, label="$v_r/|\mathbf{v}|$")
    #plt.plot(rad_dist,rad_dist*0.0+0.00017*2, lw=3)

    #plt.plot(rad_dist,np.abs(V_phi),"o", label=time*0.01)
    #

#rad_pressure(14500000,80000, 10000)

plt.ylabel("pressure")
plt.xlabel("distance from center of mass")
plt.legend(frameon=False)
plt.show()