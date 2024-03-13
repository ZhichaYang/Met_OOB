# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 10:33:49 2024
气象数据处理类
@author: Zhichao Yang
"""

"""
最后有调用例子
"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import cartopy.mpl.ticker as cticker 
from scipy.stats.mstats import ttest_ind
from scipy.stats import pearsonr
from scipy import signal
from eofs.standard import Eof
from scipy.stats import linregress
from metpy import calc
from cartopy.util import add_cyclic_point
import copy
# 读取mat文件数据
import scipy.io as scio



class MetDataProcess:
    """此类用于文件读取和数据处理"""
    def __init__(self, path,
                 variable, longitude, latitude, time, level='',
                 time_start="1960-01-01", time_end="2020-12-31", lev_range=[1000,1000],
                 missvalue=100, longitude_adjusted=False):
        """
        path: nc文件的路径
        variable: nc文件中这个变量对应的文件名
        longitude: nc文件中经度对应的文件名
        latitude: nc文件中纬度对应的文件名
        time: nc文件中时间变量对应的文件名
        level: nc文件中高度变量对应的文件名，如果没有相应变量(如SST)，可不写
        time_start, time_end:为要选取的研究时间跨度，格式为"2003-08-11"
        lev_range:为要选取的高度的范围，为一列表，如[1000,200]表明要提取1000hPa到200hPa高度的数据
        missvalue: 要将变量中大于多少的值设为nan
        longitude_adjusted:有的nc文件中经度是-180-180的，有的是0-360的，这里统一使用0-360的，如果要读取
        的nc文件是-180到180的，需要将其设为True，将经度改成0-360的
        """
        
        dataset = xr.open_dataset(path, engine='netcdf4')
        
        if longitude_adjusted:
            
            lon_name = longitude  # whatever name is in the data
            dataset['longitude_adjusted'] = xr.where(dataset[lon_name] < 0, dataset[lon_name]%360,\
                                       dataset[lon_name])
            dataset = (
                    dataset
                    .swap_dims({lon_name: 'longitude_adjusted'})
                    .sel(**{'longitude_adjusted': sorted(dataset.longitude_adjusted)})
                    .drop(lon_name))
            dataset = dataset.rename({'longitude_adjusted': lon_name})
        
        self.lon = dataset[longitude]
        self.lat = dataset[latitude]
        self.time = dataset[time].loc[time_start:time_end]
        
        ndim = len(dataset[variable].shape)
        lev_start = min(lev_range)
        lev_end = max(lev_range)
        
        if ndim==3:
            self.data = dataset[variable].loc[time_start:time_end, :, :]
            
        elif ndim==4:
            
            self.data = np.squeeze(dataset[variable].loc[time_start:time_end, lev_start:lev_end, :, :])
            self.lev = dataset[level].loc[lev_start:lev_end]    
            
        
        # 缺失值转变
        self.data.values[np.abs(self.data)>missvalue] = np.nan      
        
    def anomaly(self):
        
        """
        对原始数据求异常值
        这里不返回相应的值了，直接创建了相应的属性，
        需要即通过 实例名.data_a， 实例名.data获取
        """
        
        data_a=copy.deepcopy(self.data)
        for i in range(12):
            data_a.loc[self.time.dt.month==i+1]=signal.detrend(
                self.data.loc[self.time.dt.month==i+1], axis = 0, type = 'constant')
            
        self.data_a = data_a
        return data_a
    
    def detrend(self):
        
        """
        对原始和异常数据分别去趋势
        这里不返回相应的值了，直接创建了相应的属性，
        需要即通过 实例名.de_data_a， 实例名.de_data获取
        """
        
        nansy=self.data.isnull()
        de_data_a = copy.deepcopy(self.data_a)
        de_data_a.values=de_data_a.fillna(0)
        for i in range(12):
            de_data_a.loc[self.time.dt.month==i+1]=signal.detrend(
                de_data_a.loc[self.time.dt.month==i+1], axis = 0, type = 'linear')
        de_data_a.values[nansy]=np.nan
        
        de_data = copy.deepcopy(self.data)
        de_data.values=de_data.fillna(0)
        for i in range(12):
            de_data.loc[self.time.dt.month==i+1]=signal.detrend(
                de_data.loc[self.time.dt.month==i+1], axis = 0, type = 'linear')
        de_data.values[nansy]=np.nan
        for i in range(12):
            de_data.loc[self.time.dt.month==i+1] += self.data.loc[self.time.dt.month==i+1].mean(axis=0)
        
        self.de_data_a = de_data_a
        self.de_data = de_data
               
    def SeasonMean(self, month, anom = True, detrend=True):
        
        """
        返回每年某几个特定月份平均的数据  ，比如返回每年夏季平均的数据
        返回对应的月平均数据和选取的为平均数据以及相应的年份
        由于冬季特殊性，单独考虑，但只考虑[12,1,2]的情况
        因此返回年份可能不相同，即假如是1960-0101---2020-1231期间
        [3,4,5];[6,7,8]等会返回61年的数据，即1960-2020，而[12,1,2]会返回60年的数据，其为1961年冬季(1960.12-1961.2)-2020年冬季
        要使得两者长度相等，可以通过jja[1:]调整，即相当于1960年夏季舍弃。
        
        month: 选择平均的月份，为一列表，如[6,7,8]
        anom: 是否用异常数据计算，默认为TRUE
        detrend: 是否用去趋势数据计算，默认为TRUE
        """
        
        n = len(month)
        if detrend:
            if anom:
                data_a_sel = self.de_data_a.loc[self.time.dt.month.isin(month)]
            else:
                data_a_sel = self.de_data.loc[self.time.dt.month.isin(month)]
                
        else:
            if anom:
                data_a_sel = self.data_a.loc[self.time.dt.month.isin(month)]
            else:
                data_a_sel = self.data.loc[self.time.dt.month.isin(month)]
            

        if month[0]==12:
            
            data_a_sel = np.delete(data_a_sel, [0, 1, data_a_sel.shape[0]-1], axis=0)
            data_sm = np.zeros([np.unique(self.time.dt.year).shape[0]-1, 
                                self.lat.shape[0], self.lon.shape[0]])
            t_year = np.unique(self.time.dt.year)[1:]
            for i in range(data_sm.shape[0]):
                data_sm[i] = np.nanmean(data_a_sel[i*n:i*n+n], axis=0)      

        else:
            
            data_sm = np.zeros([np.unique(self.time.dt.year).shape[0], 
                                self.lat.shape[0], self.lon.shape[0]])
            t_year = np.unique(self.time.dt.year)
            for i in range(data_sm.shape[0]):
                data_sm[i] = np.nanmean(data_a_sel[i*n:i*n+n], axis=0)
                
        return data_sm, data_a_sel, t_year
      
    def regionmean(self, data, meantype=1, lon_start=0, lon_end=360, lat_start=-90, lat_end=90):
        
        """
        返回区域平均后的数据
        data:输入的数据，可以为实例名.data_a等属性或者是季节平均后的数据
        meantype=1:表示对区域进行平均（默认）
        meantype=2:表示对纬度进行平均
        meantype=3:表示对经度进行平均
        lon_start,lon_end: 所选区域的经度开始和结束
        lat_start,lat_end: 所选区域的纬度开始和结束
        默认对全球区域平均
        """
        
        if np.shape(data)[2]==self.lat.shape[0]:
            
            if meantype==1:
                data_rm = np.nanmean(data[:,:,(self.lat>=lat_start)&(self.lat<=lat_end),:], axis=2)
                data_rm = np.nanmean(data_rm[:,:,(self.lon>=lon_start)&(self.lon<=lon_end)], axis=2)      
            elif meantype==2: 
                data_rm = np.nanmean(data[:,:,(self.lat>=lat_start)&(self.lat<=lat_end),:], axis=2)  
            elif meantype==3:     
                data_rm = np.nanmean(data[:,:,:,(self.lon>=lon_start)&(self.lon<=lon_end)], axis=3)
        
        elif np.shape(data)[1]==self.lat.shape[0]:
            
            if meantype==1:
                data_rm = np.nanmean(data[:,(self.lat>=lat_start)&(self.lat<=lat_end),:], axis=1)
                data_rm = np.nanmean(data_rm[:,(self.lon>=lon_start)&(self.lon<=lon_end)], axis=1)            
            elif meantype==2:                
                data_rm = np.nanmean(data[:,(self.lat>=lat_start)&(self.lat<=lat_end),:], axis=1)               
            elif meantype==3:       
                data_rm = np.nanmean(data[:,:,(self.lon>=lon_start)&(self.lon<=lon_end)], axis=2)
            
        elif np.shape(data)[0]==self.lat.shape[0]:
        
            if meantype==1:
                data_rm = np.nanmean(data[(self.lat>=lat_start)&(self.lat<=lat_end),:], axis=0)
                data_rm = np.nanmean(data_rm[(self.lon>=lon_start)&(self.lon<=lon_end)], axis=0)
            elif meantype==2:               
                data_rm = np.nanmean(data[(self.lat>=lat_start)&(self.lat<=lat_end),:], axis=0)             
            elif meantype==3:      
                data_rm = np.nanmean(data[:,(self.lon>=lon_start)&(self.lon<=lon_end)], axis=1)
        
        return data_rm
    
    def yearmean(self, month=[1,2,3,4,5,6,7,8,9,10,11,12], anom = True, detrend=True):
        
        """
        返回特定几个月份的年平均数据，如返回61年平均的1月数据
        month:为选的的月份列表
        anom，detrend:同上
        """
        
        if detrend:
            if anom:
                data_a_sel = self.de_data_a.loc[self.time.dt.month.isin(month)]
            else:
                data_a_sel = self.de_data.loc[self.time.dt.month.isin(month)]
        else:
            if anom:
                data_a_sel = self.data_a.loc[self.time.dt.month.isin(month)]
            else:
                data_a_sel = self.data.loc[self.time.dt.month.isin(month)]
        n = len(month)
        data_ym = np.zeros([n, self.lat.shape[0], self.lon.shape[0]])
        for i in range(n):
            data_ym[i] = np.nanmean(data_a_sel[data_a_sel.time.dt.month==month[i]])
            
        return data_ym
    

class CompositeAnalysis:
    
    """此类用于进行合成分析"""
    
    def __init__(self, year_all, year_choose, data, lon, lat):
        
        """
        year_all:数据对应的所有年份，可通过上述MetDataProcess的SeasonMean函数得到
        year_choose:需要合成的几个特殊年份，为一列表，如[1998, 1983, 2016]
        data:原始数据，三维，分别为时间、纬度、经度。可通过上述MetDataProcess的SeasonMean函数得到
        lon，lat:原始数据对应的经度和纬度
        """
        
        self.data = data
        self.lon = lon
        self.lat = lat
        select = np.in1d(year_all, year_choose)
        self.select = select
        
    def composite(self, sig_test = True, equal_var=False):
        
        """
        对数据进行合成
        sig_test:是否对合成分析进行显著性检验
        equal_val:显著性检验默认方差不相等
        不返回数值，直接存入相应的属性，需要通过 实例名.com,实例名.p_val获取
        """
        
        self.com = np.nanmean(self.data[self.select], axis=0)
        
        if sig_test:
            _, p_val = ttest_ind(self.data[self.select], self.data, equal_var=equal_var, axis=0)
            self.p_val = p_val
            
    def complot(self, 
                pic_rows=2, pic_col=2, pic_no=1,
                central_lon=180, 
                lon_start=0, lon_end=360, lat_start=0, lat_end=90,
                vmax=1, vmin=-1,
                pictype=1,
                alpha=0.05,
                title='(a) Com'):
        
        """
        对合成图画图
        pic_rows, pic_col, pic_no:              subplot的三个参数
        central_lon:                            显示的中心纬度
        lon_start, lon_end, lat_start, lat_end: 图片显示范围，set_extent的参数
        vmax, vmin:                             colorbar的最大值最小值
        pictype:                                等于1表示画打点显著性图，等于0表示不画
        alpha:                                  打点的显著性水平，默认0.05
        title:                                  图片标题
        
        注意：其中颜色的参数我自己在文件里已经用了最常用的了，如果需要修改，直接在
        源代码里面修改'colorss' cmap.set_under和cmap.set_over这三个参数就可以了
        """
        
        fig=plt.figure()
        ax1 = plt.subplot(pic_rows, pic_col, pic_no, projection = ccrs.PlateCarree(central_longitude = central_lon))
        ax1.set_extent([lon_start, lon_end, lat_start, lat_end], crs = ccrs.PlateCarree())
        #ax1.gridlines()
        ax1.coastlines()
        ax1.set_xticks(np.arange(0, 360+40, 40),crs = ccrs.PlateCarree())
        ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax1.set_yticks(np.arange(0, 90+30, 30),crs = ccrs.PlateCarree())
        ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        levels = np.linspace(vmin, vmax, 13)
        colorss = np.array([[34,0,184], [6,0,244], [46,71,255], [106,147,255], [163,202,255], [223,241,255],
                       [255,244,229], [255,207,169], [255,154,112], [255,80,52], [250,0,2], [190,0,32]])
        cmap = mcolors.ListedColormap(colorss/255)
        cmap.set_under(np.array([42,0,127])/255)
        cmap.set_over(np.array([133,0,42])/255)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        c1 = ax1.contourf(self.lon, self.lat, self.com, levels=levels, transform=ccrs.PlateCarree(),\
                            zorder = 0, cmap = cmap, norm = norm, extend = 'both')         
        if pictype==1:                   
            c2 = ax1.contourf(self.lon, self.lat, self.p_val, levels=[0, alpha, 1], hatches=['..',None], \
                            zorder=1, colors="none", transform=ccrs.PlateCarree())
        #fig.colorbar(c1, ax = ax1, extend = 'both')#,fraction=0.032)
        ax1.set_title(title, loc = 'left', fontsize = 18)
        fig.colorbar(c1, orientation='horizontal', aspect = 50)#,format='%d',) 

        return ax1
        
    
class Regression_ArrayToFelid:
    
    """此类用于某一气候因子对整个场进行回归"""
    
    def __init__(self, array, data, lon, lat):
        
        """
        array:某气候因子的时间序列
        data: 气候场数据
        lon，lat:气候场数据对应的经度，，纬度
        """

        self.data = data
        self.array = array
        self.lon = lon
        self.lat = lat
        
    def regress_atf(self, t=1):
        
        """
        计算回归系数，相关系数和显著性p值，返回值，同时也创建了相应的属性
        t=1:表示对预报因子标准化处理（默认）
        t=2:表示不对预报因子标准化处理
        """ 
        
        a=np.zeros(self.data.shape[1:])# a:回归系数
        r=np.zeros(self.data.shape[1:])# r:相关系数
        p=np.zeros(self.data.shape[1:])# p:检验系数
        
        for i in range(self.data.shape[1]):
            for j in range(self.data.shape[2]):
                if t==1:
                    a[i,j], inte, r[i,j], p[i,j], st = linregress((self.array-np.mean(self.array))/np.std(self.array), self.data[:,i,j])
                elif t==2:
                    a[i,j], inte, r[i,j], p[i,j], st = linregress(self.array, self.data[:,i,j])
        
        self.reg_co=a
        self.coef=r
        self.p_val=p
        return a,r,p
    
    def regplot(self, 
                pic_rows=2, pic_col=2, pic_no=1,
                central_lon=180, 
                lon_start=0, lon_end=360, lat_start=0, lat_end=90,
                vmax=1, vmin=-1,
                pictype=1,
                alpha=0.05,
                title='(a) Corr'):
        
        """
        画图
        pic_rows, pic_col, pic_no:              subplot的三个参数
        central_lon:                            显示的中心纬度
        lon_start, lon_end, lat_start, lat_end: 图片显示范围，set_extent的参数
        vmax, vmin:                             colorbar的最大值最小值
        pictype:                                等于1表示画相关系数，等于0表示画回归系数
        alpha:                                  打点的显著性水平，默认0.05
        title:                                  图片标题
        
        注意：其中填色的参数我自己在文件里已经用了最常用的了，如果需要修改，直接在
        源代码里面修改'colorss' cmap.set_under和cmap.set_over这三个参数就可以了
        根据出图要不要在一张图里面，也请大家自己修改源代码
        """
        
        fig=plt.figure()
        ax1 = plt.subplot(pic_rows, pic_col, pic_no, projection = ccrs.PlateCarree(central_longitude = central_lon))
        ax1.set_extent([lon_start, lon_end, lat_start, lat_end], crs = ccrs.PlateCarree())
        #ax1.gridlines()
        ax1.coastlines()
        ax1.set_xticks(np.arange(0, 360+40, 40),crs = ccrs.PlateCarree())
        ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax1.set_yticks(np.arange(0, 90+30, 30),crs = ccrs.PlateCarree())
        ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        levels = np.linspace(vmin, vmax, 13)
        colorss = np.array([[34,0,184], [6,0,244], [46,71,255], [106,147,255], [163,202,255], [223,241,255],
                       [255,244,229], [255,207,169], [255,154,112], [255,80,52], [250,0,2], [190,0,32]])
        cmap = mcolors.ListedColormap(colorss/255)
        cmap.set_under(np.array([42,0,127])/255)
        cmap.set_over(np.array([133,0,42])/255)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        if pictype==1:    
            c1 = ax1.contourf(self.lon, self.lat, self.coef, levels=levels, transform=ccrs.PlateCarree(),\
                            zorder = 0, cmap = cmap, norm = norm, extend = 'both')         
        elif pictype==0:         
            c1 = ax1.contourf(self.lon, self.lat, self.reg_co, levels=levels, transform=ccrs.PlateCarree(),\
                            zorder = 0, cmap = cmap, norm = norm, extend = 'both')          
        c2 = ax1.contourf(self.lon, self.lat, self.p_val, levels=[0, alpha, 1], hatches=['..',None], \
                        zorder=1, colors="none", transform=ccrs.PlateCarree())
        #fig.colorbar(c1, ax = ax1, extend = 'both')#,fraction=0.032)
        ax1.set_title(title, loc = 'left', fontsize = 18)
        fig.colorbar(c1, orientation='horizontal', aspect = 50)#,format='%d',) 

        return ax1


"""
其实以上代码只实现了一些基本的功能，只能用于合成、相关、回归分析。也只能应用于月数据。
你可以自己再往上面加功能或者编写独属于自己的类，不断地完善，从而提供科研效率。
希望能起到抛转引玉的作用
"""

"""
调用举例
将该文件放在同一个文件夹下并在新文件中写入
import import DataProcessing as dp即可
"""

# import DataProcessing as dp
#%% 读取文件(用的HadLey中心SST数据和JRA55的U数据)
# path="D:\\keyan\\ENSO and Precipitation\\Dataset\\HadISST_sst.nc";    
# data_sst=dp.MetDataProcess(path, 'sst', 'longitude', 'latitude', 'time', time_start="1960-01-01", time_end="2020-12-31", longitude_adjusted=True)
# data_sst.anomaly()
# data_sst.detrend()
# sst_djf,_,year_all=data_sst.SeasonMean([12, 1, 2])
# nino34_djf= data_sst.regionmean(sst_djf, lon_start=190, lon_end=240, lat_start=-5, lat_end=5)
    
    
# path='D:\keyan\ENSO and Precipitation\Dataset\JRA55\\anl_p125.033_ugrd.195801_202012.mon.nc'
# data_u200=dp.MetDataProcess(path, 'UGRD_GDS0_ISBL_S123', 'g0_lon_3', 'g0_lat_2', 'initial_time0_hours', level='lv_ISBL1',lev_range=[200])
# data_u200.anomaly()
# data_u200.detrend()
# u0_djf,_,_=data_u200.SeasonMean([12,1,2],anom=False,detrend=True)

#%% 选取厄尔尼诺并做合成分析

# #根据标准化的冬季nino3.4指数选取
# ninoch = (nino34_djf-np.mean(nino34_djf))/np.std(nino34_djf)>0.75
# nino_year = year_all[ninoch]
# ninocom = dp.CompositeAnalysis(year_all, nino_year, sst_djf, data_sst.lon, data_sst.lat)
# ninocom.composite()
# ninocom.complot(pic_rows=1,pic_col=1, title='ninocom',pictype=1,lat_start=-90)

#%% 相关，nino3.4和U200的相关

# data_u200=dp.Regression_ArrayToFelid(nino34_djf, u0_djf, data_u200.lon, data_u200.lat)
# data_u200.regress_atf()
# ax1=data_u200.regplot(pic_rows=1,pic_col=1,lat_start=-90)





















