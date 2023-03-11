import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import csv

import tkinter
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg





class Scribble:
	def __init__(self):
		self.master = tkinter.Tk()
		self.master.protocol("WM_DELETE_WINDOW", self.delete_window)
		#このプログラムの名前を取得
		self.programTitle = os.path.abspath(__file__)[:-3]
		#このプログラムで取得するデータのフォルダを作成
		os.makedirs(self.programTitle, exist_ok = True)
		self.fileOpen()
		self.window = self.create_window();
		self.default_fontfamily = "Yu Gothic UI"
		self.default_fontsize = 10
		self.__create_menu()
		# self.net, self.dataset = eval_ex.eval_main()
		
	def run(self):
		self.root.mainloop()
		
	def create_window(self):
		#メニューバーの作成
		self.menubar = tkinter.Menu(self.master)
		self.filemenu = tkinter.Menu(self.menubar)
		self.filemenu.add_command(label = '開く', command = self.fileChange)
		self.menubar.add_cascade(label = "file", menu = self.filemenu)
		self.master.config(menu = self.menubar)
		
		#フレームの作成
		heights, widths  = self.videoShape
		
		self.root = ttk.Frame(self.master, width = widths*1.8, height = heights*1.8)
		self.root.pack()
		
		self.makeMainFrame()
		self.makeCellFrame()
		self.makeGraphFrame()
		self.makeDataFrame()
		self.master.focus_set()


		return self.master;

	def delete_window(self):
		self.master.destroy()
		sys.exit()

	def makeMainFrame(self):
		heights, widths  = self.videoShape
		self.mainFrame = ttk.Frame(self.root, width = widths*1.2, height = heights*1.2)
		self.mainFrame.place(x = 0, y = 0)
		
		self.movieCanvas = tkinter.Canvas(self.mainFrame, width = widths, height = heights)
		self.movieCanvas.place(x = widths*0.1, y = heights*0.1)
		movie_image = cv2.imread(self.videoImage_path + '/image_' +str(0).zfill(len(str(self.videoLength)))+'.png')
		#print(movie_image)
		
		
		self.processImage_list = []
		self.processName_list = []
		self.processOption_list = []
		self.processButtonPropaty = []
		self.processImage_list.append([copy.deepcopy(movie_image)])
		self.processName_list.append(['original'])
		self.processOption_list.append([[]])
		self.movie_image = self.makeTkImage(movie_image, self.master)
		self.movieCanvas.create_image(0, 0, image=self.movie_image, anchor='nw')
		self.movieCanvas.bind('<ButtonPress-1>', self.onClicked)
		self.movieCanvas.bind('<Motion>', self.onMotion)
		#self.movieCanvas.bind('<ButtonPress-2>', self.cellSave)
		self.movieCanvas.bind('<ButtonPress-2>', self.__do_popup)
		self.movieCanvas.bind('<MouseWheel>', self.onWheel)
		
		self.master.bind('<KeyPress>', self.keyPress)
		
		self.scaleFrame = ttk.Frame(self.mainFrame, width = widths*1.2, height = heights *0.05)
		self.scaleFrame.place(x = 0, y = heights*1.15)
		self.scaleButton1 = tkinter.Button(self.scaleFrame, width = 1, height = 1, text = '>', command = self.videoStart)
		self.scaleButton1.pack(side = LEFT)
		self.scaleButton2 = tkinter.Button(self.scaleFrame, width = 1, height = 1, text = '||', command = self.videoStop)
		self.scaleButton2.pack(side = LEFT)
		self.sc_val = DoubleVar()
		self.sc = ttk.Scale(
		self.scaleFrame,
		variable=self.sc_val,
		orient=HORIZONTAL,
		length=widths,
		from_=0,
		to=self.videoLength,
		command= self.scale)
		self.sc.pack(side = LEFT)
		
		
	def makeCellFrame(self):
		h, w = self.videoShape
		self.cellFrame = ttk.Frame(self.root, width = w*0.6, height = h*0.6)
		self.cellFrame.place(x = w*1.2, y = h*1.2)
		self.cellFrameCanvas = tkinter.Canvas(self.cellFrame, width= w*0.6, height = h*0.6)
		self.cellFrameCanvas.pack()
		
		
	def makeGraphFrame(self):
		heights, widths  = self.videoShape
		pixToInch = 128*1.25
		heightsInch = heights/pixToInch
		widthsInch = widths/pixToInch
		self.graphFrame = ttk.Frame(self.root, width = widths*0.6, height = heights*1.2)
		self.graphFrame.place(x = widths*1.2, y = 0)
		
		self.roundFrame = ttk.Frame(self.graphFrame, width = widths*0.6, height= heights*0.4)
		self.roundFrame.pack(side = TOP)
		print(widthsInch*0.6, heightsInch*0.4)
		self.roundFig = plt.figure(figsize=(widthsInch*0.6, heightsInch*0.4))
		self.roundGraph = self.roundFig.add_subplot(111)
		self.roundFigCanvas = FigureCanvasTkAgg(self.roundFig, master=self.roundFrame)  # Generate canvas instance, Embedding fig in root
		self.roundFigCanvas.get_tk_widget().pack(side = LEFT,padx = 0)
		self.roundFigCanvas.get_tk_widget().bind('<ButtonPress-1>', self.graphView2)
		
		self.moveFrame = ttk.Frame(self.graphFrame, width = widths*0.6, height= heights*0.8)
		self.moveFrame.pack(side = TOP)
		self.moveFig = plt.figure(figsize=(widthsInch*0.6, heightsInch*0.8))
		self.moveGraph = self.moveFig.add_subplot(111)
		self.moveFigCanvas = FigureCanvasTkAgg(self.moveFig, master=self.moveFrame)  # Generate canvas instance, Embedding fig in root
		self.moveFigCanvas.get_tk_widget().pack(side = LEFT,padx = 0)
		self.graphview = 0
		self.moveFigCanvas.get_tk_widget().bind('<ButtonPress-1>', self.graphView2)
		
	def makeDataFrame(self):
		h, w = self.videoShape
		self.dataFrame = ttk.Frame(self.root, width = w*1.2, height = h*0.6)
		self.dataFrame.place(x = 0, y = h*1.2)
		fonts=("MSゴシック", "30")
		texts = ['時刻', '中心座標', 'サイズ', '周囲長', '円形度']
		labelPlaces = [[0, 0], [0, w*1.2*(1/3)], [h*0.6*0.5, w*1.2*(0/3)], [h*0.6*0.5, w*1.2*(1/3)], [h*0.6*0.5, w*1.2*(2/3)]]
		textPlaces = [[0, 90], [0, w*1.2*(1/3)+150], [h*0.6*0.5, w*1.2*(0/3)+120], [h*0.6*0.5, (w*1.2*(1/3)+120)], [(h*0.6*0.5), (w*1.2*(2/3)+120)]]
		self.label = []
		self.label_data = []
		for i in range(len(texts)):
			self.label.append([tkinter.Label(master = self.dataFrame, text=texts[i],  font=fonts)])
			self.label[i][0].place(y = labelPlaces[i][0], x = labelPlaces[i][1])
			self.label_data.append([tkinter.Entry(master = self.dataFrame, width=8, font = fonts)])
			self.label_data[i][0].place(y = textPlaces[i][0], x = textPlaces[i][1])
		

	def fileOpen(self):
		#動画ファイルの読み込み
		if(1==2):#開発用データ
			self.file = 'macro_30fps.avi'
		else:#ファイル選択のホップアップ
			iDir = os.path.abspath(os.path.dirname(__file__))
			self.file = tkinter.filedialog.askopenfilename(initialdir = iDir)
			if self.file == "":
				return "break"
		self.dir_path = self.programTitle + '/'+os.path.split(self.file)[1][:-4]
		print(self.dir_path)
		
		
		#動画の大きさ, 長さの取得
		cap = cv2.VideoCapture(str(self.file))
		self.videoShape = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))]
		self.videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		
		
		#動画のコンバート
		self.videoImage_path = self.dir_path +'/originvideo'
		if(os.path.exists(self.dir_path+'/originvideo') == False):
			os.makedirs(self.videoImage_path, exist_ok = True)
			digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
			for i in range(self.videoLength):
				ret, frame = cap.read()
				img_path = self.videoImage_path + '/image_' +str(i).zfill(digit)+'.png'
				cv2.imwrite(img_path, frame)
				
		self.videoFlag = 0
		self.sc_val2 = 0
		self.cellFlag = False
		self.motionFlag = False
		self.rect = False
		self.videoStops = 0
		self.targetArea2 = np.zeros((4))
		self.targetArea3 = [30, 30, 30, 30]
		self.cellRange = np.zeros((4))


	def fileChange(self):
		self.fileOpen
		self.window = self.create_window();
		
		
	def scale(self, movieTime):
		movieTime = int(float(movieTime))
		h, w = self.videoShape
		self.sc_val2 = movieTime
		digit = len(str(self.videoLength))
		img_path = self.videoImage_path + '/image_' +str(int(movieTime)).zfill(digit)+'.png'
		frame = cv2.imread(img_path)
		img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		image_w = copy.deepcopy(frame)
		#print('a')
		if(self.cellFlag == 1):
			x1 = int(self.cellRange[1])
			y1 = int(self.cellRange[0])
			x2 = int(self.cellRange[1] + self.cellRange[3])
			y2 = int(self.cellRange[0] + self.cellRange[2])
			
			X1 = int(max(x1-5, 0))
			X2 = int(min(x2+5, h-1))
			#print(y1, x1)
			Y1 = int(max(y1-5, 0))
			Y2 = int(min(y2+5, w-1))
			
			imgTarget = copy.deepcopy(img_gray[Y1:Y2, X1:X2])
			imgPickup = copy.deepcopy(image_w[Y1:Y2, X1:X2])
			print("--imgSegmen--")
			# imgSegmen, maskSegmen = eval_ex.evalimage_ex(self.net, imgPickup)
			# cv2.imwrite(self.cellEtc_path+ '/image_' +str(int(movieTime)).zfill(digit)+'_etc1.png', imgSegmen)
			# if(len(maskSegmen)):
			# 	cv2.imwrite(self.cellEtc_path+ '/image_' +str(int(movieTime)).zfill(digit)+'_etc2.png', np.array(maskSegmen.cpu().detach(), dtype = "u1"))
			h, w = np.shape(imgTarget)
			
			checkMap = np.zeros((h, w))
			image2 = copy.deepcopy(imgTarget)
			for i in range(h):
				for j in range(w):
					if(image2[i][j]>= 133 or image2[i][j]<=110):
						image2[i][j] =255
					else:
						image2[i][j] = 0
			image2 = np.array(image2, dtype = 'u1')
			cv2.imwrite(self.cellEtc_path+ '/image_' +str(int(movieTime)).zfill(digit)+'_etc.png', image2)
			area = cv2.connectedComponentsWithStats(image2)[2]
			#print(area)
			pointList_2 = []
			for a  in range(len(area)-1):
				checkMap = np.zeros((h, w))
				h2, w2 = int(area[a+1][3]), int(area[a+1][2])
				xx = int(area[a+1][0])
				yy = int(area[a+1][1])
				xxx = int(area[a+1][0] +area[a+1][2] -1)
				yyy = int(area[a+1][1] + area[a+1][3] -1)
				
				xx = max(0, xx-1)
				yy = max(0, yy-1)
				xxx = min(w-1, xxx+1)
				yyy = min (h-1, yyy+1)
				h2, w2 = yyy-yy, xxx-xx
				outerList = []
				for j in range(w2):
					outerList.append([0+yy, j+xx])
				for i in range(1, h2):
					outerList.append([i+yy, w2-1+xx])
				for j in range(w2-2, -1, -1):
					outerList.append([h2-1+yy, j+xx])
				for i in range(h2-2, 0, -1):
					outerList.append([i+yy, 0+xx])
				pointList = []
				while outerList:
					y, x = outerList[0]
					outerList.pop(0)
					if(checkMap[y][x]==0):
						pointList.append([y, x])
						pointList2 = []
						pointList2.append([y, x])
						pointList2_2 = []
						pointList2_2.append([y, x])
						checkMap[y][x] = 255
						while pointList2:
							y2, x2 = pointList2[0]
							pointList2.pop(0)
							if(y2>yy and checkMap[y2-1][x2]==0):
								if(imgTarget[y2][x2] == imgTarget[y2-1][x2]):
									pointList.append([y2-1, x2])
									pointList2.append([y2-1, x2])
									pointList2_2.append([y2-1, x2])
									checkMap[y2-1][x2] = 255
								elif(imgTarget[y2][x2] <= imgTarget[y2-1][x2]):
									outerList.append([y2-1, x2])
							if(x2>xx and checkMap[y2][x2-1]==0):
								if(imgTarget[y2][x2] == imgTarget[y2][x2-1]):
									pointList.append([y2, x2-1])
									pointList2.append([y2, x2-1])
									pointList2_2.append([y2, x2-1])
									checkMap[y2][x2-1] = 255
								elif(imgTarget[y2][x2] <= imgTarget[y2][x2-1]):
									outerList.append([y2, x2-1])
							if(y2<yyy-1 and checkMap[y2+1][x2]==0  ):
								if(imgTarget[y2][x2] == imgTarget[y2+1][x2]):
									pointList.append([y2+1, x2])
									pointList2.append([y2+1, x2])
									pointList2_2.append([y2+1, x2])
									checkMap[y2+1][x2] = 255
								elif(imgTarget[y2][x2] <= imgTarget[y2+1][x2]):
									outerList.append([y2+1, x2])
							if(x2<xxx-1 and checkMap[y2][x2+1]==0):
								if(imgTarget[y2][x2] == imgTarget[y2][x2+1]):
									pointList.append([y2, x2+1])
									pointList2.append([y2, x2+1])
									pointList2_2.append([y2, x2+1])
									checkMap[y2][x2+1] = 255
								elif(imgTarget[y2][x2] <= imgTarget[y2][x2+1]):
									outerList.append([y2, x2+1])
						pointList_2.append([pointList2_2])
					#plt.imshow(checkMap, cmap = 'gray')
					#plt.show()
			#outerList = copy.deepcopy(pointList_2)
			h, w = np.shape(imgTarget)
			#plt.imshow(checkMap, cmap = 'gray')
			#plt.show()
			checkMap = np.zeros((h, w))
			for a  in range(len(area)-1):
				checkMap[area[a+1][1]:area[a+1][1]+area[a+1][3], area[a+1][0]:area[a+1][0]+area[a+1][2]] = 255
			for i in pointList_2:
				#print(i[0])
				for j in i[0]:
					checkMap[j[0]][j[1]] = 0
				#plt.imshow(checkMap, cmap = 'gray')
				#plt.show()
			img_black = copy.deepcopy(checkMap)
			#img_black = 255 - img_black
			img_black = np.array(img_black, dtype = 'u1')
			cv2.imwrite(self.cellEtc_path+ '/image_' +str(int(movieTime)).zfill(digit)+'_etc2.png', img_black)
			contours = cv2.connectedComponentsWithStats(img_black)[1]
			area = cv2.connectedComponentsWithStats(img_black)[2]
			area_sort = area[np.argsort(area[:, 4])[::-1]]
			img_black = np.zeros((h, w))
			for i in range(h):
				for j in range(w):
					if(area[contours[i][j]][4] == area_sort[1][4]):
						img_black[i][j] = 255
			for m in range(1):
				if(m == 0):
					img_black = np.array(img_black, dtype = "u1")
				# if(m == 1):
				# 	image_w = copy.deepcopy(frame)
				# 	img_black_2 = np.array(maskSegmen.cpu().detach(), dtype = 'u1')
				# 	if(img_black_2.shape[0]>0):
				# 		img_black = np.reshape(img_black_2[0], [np.shape(image2)[0], np.shape(image2)[1]])
				# 		img_black = np.array(img_black*255, dtype = "u1")

				# img_black = np.array(cv2.threshold(img_black, 128, 255, cv2.THRESH_BINARY)[1], dtype = "u1")
				cv2.imwrite(self.cellEtc_path+ '/image_' +str(int(movieTime)).zfill(digit)+'_etc2.png', img_black)
				#plt.imshow(img_black, cmap = 'gray')
				#plt.show()
				contours = cv2.connectedComponentsWithStats(img_black)[1]
				area = cv2.connectedComponentsWithStats(img_black)[2]
				cellCenter = cv2.connectedComponentsWithStats(img_black)[3][1]
				cellBoxCenter = [area[1][0]+0.5*(area[1][2]), area[1][1]+0.5*(area[1][3])]
				menseki = area[1][4]
				roundLength = 0
				cellCenter[0] += x1
				cellCenter[1] += y1
				cellBoxCenter[0] += x1
				cellBoxCenter[1] += y1
				self.cellRange_list = list(self.cellRange_list)
				self.cellRange_list.append(cellCenter)
				self.cellBoxRange_list = list(self.cellBoxRange_list)
				self.cellBoxRange_list.append(cellBoxCenter)
				self.label_data[1][0].delete(0, tkinter.END)
				self.label_data[1][0].insert(tkinter.END, '('+str(int(cellCenter[0]))+','+str(int(cellCenter[1])) +')')
				if(len(self.cellTime) > 1):
					move = ((self.cellRange_list[-1][0] - self.cellRange_list[-2][0])**2 + (self.cellRange_list[-1][1] - self.cellRange_list[-2][1])**2)**0.5
					move = int(move)
					self.cellMove_list.append(move)
				img_black = np.zeros((h, w))
				que = []
				freemanChain = [[-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1]]
				chainNum = -8
				chainPre = 0
				flag = False
				for i in range(h):
					for j in range(w):
						if flag:
							break
						if(contours[i][j] == 1):
							first = [i, j]
							flag = True
				flag = False
				xx = first[1]
				yy = first[0]
				contour_points = [xx, yy]
				for i in range(h):
					for j in range(w):
						if flag:
							break
						flag2 = False
						for k in range(len(freemanChain)) :
							if(flag2):
								break
							y = yy + freemanChain[k + chainNum][0]
							x = xx + freemanChain[k + chainNum][1]
							if(x>=0 and x< w and y>= 0 and y< h):
								if(contours[y][x]!=0):
									flag2 = True
									yy = y
									xx = x
									chainNum = k + chainNum -3
									if(chainNum<-8):
										chainNum += 8
									if(chainNum>0):
										chainNum -= 8
									que.append([y, x])
									img_black[y][x] = 255
									image_w[y+Y1][x+X1] = [0, 255, 0]
									if((-1*chainNum)%2 == 0):
										roundLength += 1
									else:
										roundLength += 1.406
									if([yy, xx]==first):
										flag = True
									else:
										contour_points.extend([xx, yy])
										
				
				img_black = np.array(img_black, dtype = 'u1')
				image = copy.deepcopy(image_w[Y1:Y2, X1:X2])
				if(m == 0):
					cv2.imwrite(self.cellContours_path+ '/contours_' +str(int(movieTime)).zfill(digit)+'.png', image)
					cv2.imwrite(self.cellContours_path_1+ '/contours_' +str(int(movieTime)).zfill(digit)+'_my.png', image)
					# self.cellRoundRate2.append(min(4*3.14*menseki/(roundLength)**2, 1))
				if(m == 1):
					cv2.imwrite(self.cellContours_path+ '/contours_' +str(int(movieTime)).zfill(digit)+'_yolact.png', image)
					cv2.imwrite(self.cellContours_path_2+ '/contours_' +str(int(movieTime)).zfill(digit)+'_yolact.png', image)
			cv2.imwrite(self.cellImage_path+ '/image_' +str(int(movieTime)).zfill(digit)+'.png', imgTarget)
			cv2.imwrite(self.cellEdge_path+ '/edge_' +str(int(movieTime)).zfill(digit)+'.png', img_black)
			# cv2.imwrite(self.cellContours_path+ '/contours_' +str(int(movieTime)).zfill(digit)+'.png', image)
			self.cellArea.append(menseki)
			self.label_data[2][0].delete(0, tkinter.END)
			self.label_data[2][0].insert(tkinter.END, str(menseki))
			self.cellRound.append(roundLength)
			self.label_data[3][0].delete(0, tkinter.END)
			self.label_data[3][0].insert(tkinter.END, str(int(roundLength)))
			self.cellRoundRate.append(min(4*3.14*menseki/(roundLength)**2, 1))
			self.label_data[4][0].delete(0, tkinter.END)
			self.label_data[4][0].insert(tkinter.END, str(min(4*3.14*menseki/(roundLength)**2, 1))[:5])
			self.cellContours.append([str(int(movieTime)).zfill(digit), [h, w, [area[1][0], area[1][1], area[1][2], area[1][3], area[1][4]]], contour_points])
			#decide fx
			fx = int(min((self.videoShape[0]/image.shape[0]), (self.videoShape[1]/image.shape[1]))*0.5)
			image= cv2.resize(image, dsize =None, fx=fx, fy=fx)
			self.cellFrameImage = self.makeTkImage(image, self.master)
			self.cellFrameCanvas.create_image(0.3*self.videoShape[1], 0.3*self.videoShape[0], image = self.cellFrameImage)
			#chanVace
			'''
			img = copy.deepcopy(imgTarget)
			img = np.array(img, dtype = 'u1')
			h, w = np.shape(img)
			mask = np.zeros(img.shape)
			mask[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)] = 1
			seg, phi, its = chanvace2.chanvese(img, mask, max_its=1000, display=False, alpha=1.0)
			for i in range(h):
				for j in range(w):
					if(i>0 and phi[i-1][j]*phi[i][j]<0):
						if(phi[i][j]<0):
							print('True')
							image_w[i+y1][j+x1] = [0, 255, 0]
					if(j>0 and phi[i][j-1]*phi[i][j]<0):
						if(phi[i][j]<0):
							print('True')
							image_w[i+y1][j+x1] = [0, 255, 0]
					if(i<h-1 and phi[i+1][j]*phi[i][j]<0):
						if(phi[i][j]<0):
							print('True')
							image_w[i+y1][j+x1] = [0, 255, 0]
							
					if(j<w-1 and phi[i][j+1]*phi[i][j]<0):
						if(phi[i][j]<0):
							print('True')
							image_w[i+y1][j+x1] = [0, 255, 0]
			phi = np.array(phi, dtype = 'u1')
			'''
			#cv2.imwrite(self.cellImage_path+ '/image_' +str(int(movieTime)).zfill(digit)+'-2.png', image_w)
			self.Apdate2(image_w)
					
						
		image_w = np.array(image_w, dtype = 'u1')
		self.Apdate(image_w)
			
	def Apdate(self, image_w):
		h, w = self.videoShape
		self.movie_image = self.makeTkImage(image_w, self.master)
		self.movieCanvas.create_image(0, 0, image=self.movie_image, anchor='nw')
		self.rect = self.movieCanvas.create_rectangle(self.targetArea2[1], self.targetArea2[0], self.targetArea2[3], self.targetArea2[2], tag = 'rect')

	def Apdate2(self, image):
		self.roundGraph.remove()
		self.roundGraph = self.roundFig.add_subplot(111)
		a = 150*(int(len(self.cellTime)/150)+1)
		self.roundGraph.plot(a, 1)
		self.roundGraph.plot(a, 0)
		self.roundGraph.plot(self.cellTime, self.cellRoundRate)
		self.roundFigCanvas.draw()
		
		h, w , z= np.shape(image)
		self.moveGraph.remove()
		self.moveGraph = self.moveFig.add_subplot(111)
		self.cellRange_list = np.array(self.cellRange_list)
		y1 = max(min(self.cellRange_list[:, 0])-25, 0)
		x1 = max(min(self.cellRange_list[:, 1])-25, 0)
		y2 = min(max(self.cellRange_list[:, 0])+25, h)
		x2 = min(max(self.cellRange_list[:, 1])+25, w)
		'''
		print('y1', y1)
		print('x1', x1)
		print('y2', y2)
		print('x2', x2)
		'''
		self.moveGraph.plot(x1, y1)
		self.moveGraph.plot(x2, y2)
		self.moveGraph.plot(self.cellRange_list[:, 0], self.cellRange_list[:, 1])
		self.moveGraph.invert_yaxis()
		self.moveFigCanvas.draw()
		
	def onClicked(self, event):
		if(self.cellFlag == 1):
			#print(self.videoStops)
			if(self.videoStops == 1):
				self.videoStops = 0
				self.onClicked2()
				self.videoStops = 1
			self.videoStops += 1
			self.videoStops = self.videoStops % 2
				
			
		else:
			self.videoStops = 0
			if(self.sc_val2 < self.videoLength):
				h, w = self.videoShape
				self.cellRange_list = []
				self.cellBoxRange_list = []
				self.cellMove_list = []
				self.cellArea = []
				self.cellRound = []
				self.cellRoundRate = []
				self.cellRoundRate2 = []
				self.cellContours = []
				self.cellTime = []
				print(len(self.cellRange_list), "*")
				self.cellTime.append(int(self.sc_val2))
				self.label_data[0][0].insert(tkinter.END, str(self.sc_val2))
				self.cellFlag = 1
				
				self.cellRange[0] = int(self.targetArea2[0])
				self.cellRange[1] = int(self.targetArea2[1])
				self.cellRange[2] = int(self.targetArea2[2] - self.targetArea2[0])
				self.cellRange[3] = int(self.targetArea2[3] - self.targetArea2[1])
				
				self.cell_path = self.dir_path + '/time_' + str(self.sc_val2) + 'place_(' + str(self.cellRange[0]) + ',' + str(self.cellRange[1]) + ')'
				self.cellImage_path = self.cell_path + '/image'
				self.cellEdge_path = self.cell_path + '/edge'
				self.cellContours_path = self.cell_path + '/contours'
				self.cellContours_path_1 = self.cell_path + '/contours_1'
				self.cellContours_path_2 = self.cell_path + '/contours_2'				
				self.cellEtc_path = self.cell_path + '/etc'
				
				os.makedirs(self.cellImage_path, exist_ok = True)
				os.makedirs(self.cellEdge_path, exist_ok = True)
				os.makedirs(self.cellContours_path, exist_ok = True)
				os.makedirs(self.cellContours_path_1, exist_ok = True)
				os.makedirs(self.cellContours_path_2, exist_ok = True)
				os.makedirs(self.cellEtc_path, exist_ok = True)
				self.scale(self.sc_val2)
				self.onClicked2()
				#self.root.after(10, self.onClicked2)
				
	def onClicked2(self):
		if(self.sc_val2 < self.videoLength and self.videoStops == 0):
			self.sc_val2 += 1
			self.cellTime.append(int(self.sc_val2))
			print(len(self.cellRange_list))
			self.label_data[0][0].delete(0, tkinter.END)
			self.label_data[0][0].insert(tkinter.END, str(self.sc_val2))
			h, w = self.videoShape
			self.cellRange[0] = int(self.targetArea2[0])
			self.cellRange[1] = int(self.targetArea2[1])
			self.cellRange[2] = int(self.targetArea2[2] - self.targetArea2[0])
			self.cellRange[3] = int(self.targetArea2[3] - self.targetArea2[1])
			self.scale(self.sc_val2)
			self.root.after(10, self.onClicked2)
			
	def cellSave(self):
		print('Saved')
		end_time = int(self.sc_val2)
		time_length = end_time - self.cellTime[0]
		digit = len(str(self.videoLength))
		self.cellFlag = 0
		wb = openpyxl.Workbook()
		ws = wb['Sheet']
		ws.cell(row = 1, column = 1).value = 'time'
		ws.cell(row = 1, column = 2).value = '面積'
		ws.cell(row = 1, column = 3).value = '周囲長'
		ws.cell(row = 1, column = 4).value = '円形度'
		ws.cell(row = 1, column = 5).value = '重心座標x'
		ws.cell(row = 1, column = 6).value = '重心座標y'
		ws.cell(row = 1, column = 7).value = '矩形座標x'
		ws.cell(row = 1, column = 8).value = '矩形座標y'

		
		for i in range(time_length+1):
			ws.cell(row = i+2, column = 1).value = self.cellTime[i]
			ws.cell(row = i+2, column = 2).value = self.cellArea[i]
			ws.cell(row = i+2, column = 3).value = self.cellRound[i]
			ws.cell(row = i+2, column = 4).value = self.cellRoundRate[i]
			ws.cell(row = i+2, column = 5).value = self.cellRange_list[i][0]
			ws.cell(row = i+2, column = 6).value = self.cellRange_list[i][1]
			ws.cell(row = i+2, column = 7).value = self.cellBoxRange_list[i][0]
			ws.cell(row = i+2, column = 8).value = self.cellBoxRange_list[i][1]
			# ws.cell(row = i+2, column = 9).value = self.cellRoundRate2[i]
		
		
			
		ws.cell(row = i+4, column = 4).value = '平均値(円形度)'
		ws.cell(row = i+4, column = 5).value = '標準偏差(円形度)'
		ws.cell(row = i+5, column = 4).value = np.average(self.cellRoundRate)
		ws.cell(row = i+5, column = 5).value = np.std(self.cellRoundRate)
		
		wb.save(self.cell_path +'/cell_analyze.xlsx')
		with open(self.cell_path +'/cell_contours.csv', "w")as f:
			writer = csv.writer(f)
			writer.writerows(self.cellContours)
		image = cv2.imread(self.videoImage_path + '/image_' +str(int(end_time)).zfill(digit)+'.png')
		self.Apdate(image)
		
		
		
		
	def onMotion(self, event):
		if(self.rect):
			self.movieCanvas.delete(self.rect)

		if(self.motionFlag == False):
			self.targetArea2[0] = event.y - self.targetArea3[0]
			self.targetArea2[1] = event.x - self.targetArea3[1]
			self.targetArea2[2] = event.y + self.targetArea3[2]
			self.targetArea2[3] = event.x + self.targetArea3[3]
			self.rect = self.movieCanvas.create_rectangle(self.targetArea2[1], self.targetArea2[0], self.targetArea2[3], self.targetArea2[2], tag = 'rect')
			self.motionFlag = True
		else:
			y = (event.y - self.targetArea3[0]) - self.targetArea2[0]
			x = (event.x - self.targetArea3[1]) - self.targetArea2[1]
			self.targetArea2[0] = event.y - self.targetArea3[0]
			self.targetArea2[1] = event.x - self.targetArea3[1]
			self.targetArea2[2] = event.y + self.targetArea3[2]
			self.targetArea2[3] = event.x + self.targetArea3[3]
			self.rect = self.movieCanvas.create_rectangle(self.targetArea2[1], self.targetArea2[0], self.targetArea2[3], self.targetArea2[2], tag = 'rect')

			#print(self.targetArea2[0], self.targetArea2[1])

	def onWheel(self, event):
		print("onWheel")
		print(int(event.delta/120))
		x = int(event.delta/120)
		h, w = self.videoShape
		# d = event.delta
		if(self.rect):
			self.movieCanvas.delete(self.rect)
		if(self.targetArea3[0]>0 and self.targetArea2[0]>0 and self.targetArea3[0]<h-1 and self.targetArea2[0]<h-1):
			self.targetArea3[0] += x * 2
			self.targetArea2[0] += x * -1
		if(self.targetArea3[1]>0 and self.targetArea2[1]>0 and self.targetArea3[1]<w-1 and self.targetArea2[1]<w-1):
			self.targetArea3[1] += x * 2
			self.targetArea2[1] += x * -1
		if(self.targetArea3[2]>0 and self.targetArea2[2]>0 and self.targetArea3[2]<h-1 and self.targetArea2[2]<h-1):
			self.targetArea3[2] += x * 2
			self.targetArea2[2] += x * 1
		if(self.targetArea3[3]>0 and self.targetArea2[3]>0 and self.targetArea3[3]<w-1 and self.targetArea2[3]<w-1):
			self.targetArea3[3] += x * 2
			self.targetArea2[3] += x * 1
		self.movieCanvas.create_image(w, h, image = self.movie_image, anchor = 'nw')
		print("target3", self.targetArea3)
		print("target2", self.targetArea2)
		self.rect = self.movieCanvas.create_rectangle(self.targetArea2[1], self.targetArea2[0], self.targetArea2[3], self.targetArea2[2])

			
	def keyPress(self, event):
		self.motionFlag = False
		key = event.keysym
		#print(key)
		h, w = self.videoShape
		if(self.rect):
			self.movieCanvas.delete(self.rect)
		
		if(key == 'w'):
			self.targetArea3[0] += 1
			self.targetArea2[0] -= 1
		if(key == 'W'):
			self.targetArea3[0] -= 1
			self.targetArea2[0] += 1
		
		if(key == 'a'):
			self.targetArea3[1] += 1
			self.targetArea2[1] -= 1
		if(key == 'A'):
			self.targetArea3[1] -= 1
			self.targetArea2[1] += 1
			
		if(key == 's'):
			self.targetArea3[2] += 1
			self.targetArea2[2] += 1
		if(key == 'S'):
			self.targetArea3[2] -= 1
			self.targetArea2[2] -= 1

		if(key == 'd'):
			self.targetArea3[3] += 1
			self.targetArea2[3] += 1
		if(key == 'D'):
			self.targetArea3[3] -= 1
			self.targetArea2[3] -= 1
		if(key == 's'+'ctrl'):
			self.cellSave(1)
		
		self.movieCanvas.create_image(w, h, image = self.movie_image, anchor = 'nw')
		self.rect = self.movieCanvas.create_rectangle(self.targetArea2[1], self.targetArea2[0], self.targetArea2[3], self.targetArea2[2])
			
		
		
	def makeTkImage(self, image, master):
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image_array = Image.fromarray(image_rgb)
		image_tk  = ImageTk.PhotoImage(image_array, master = master)
		return image_tk
	def videoStart(self):
		if(self.sc_val2 < self.videoLength and self.videoFlag == 0):
			self.sc_val2 += 1
			self.sc.set(self.sc_val2)
			self.root.after(30, self.videoStart)
			
		else:
			self.videoFlag = 0
		
	def videoStop(self):
		self.videoFlag = 1

	def graphView1(self):
		h, w = self.videoShape
		'''
		self.masterGraph = tkinter.Toplevel()
		self.rootGraph = ttk.Frame(self.masterGraph, width = w, height = h)
		self.rootGraph.pack()
		self.windowGraph = ttk.Frame(self.rootGraph, width = w, height = h)
		self.windowGraph.pack()
		'''
		self.figCanvasEx1 = tkinter.Canvas(self.mainFrame, width = w, height = h)
		self.figCanvasEx1.place(y = 0.1*h, x = 0.1*w)
		self.figCanvasEx1.bind('<Motion>', self.graphOnImageDraw)
		self.figCanvasEx1.bind('<ButtonPress-1>', self.graphOnImage)
		h, w = 0.8*h, 0.95*w
		self.figCanvasEx1.create_rectangle(0.2*w, 0.1*h, w, h)
		average = [np.average(self.cellRoundRate)]*len(self.cellRoundRate)
		std = np.std(self.cellRoundRate, keepdims = True)
		std_plus = average+std
		std_minus = average- std
		font_size = 20
		fonts=('Times New Roman', "20")
		x_p = 0
		y_p = 0
		x = 0
		y = 0
		h2 = (h*0.9)
		w2 = ((w*0.8))/(len(self.cellTime))
		for i in range(len(self.cellTime)-1):
			x_p =  w2* i +0.2*w
			y_p = h2* (1 - self.cellRoundRate[i]) + 0.1*h
			x = w2 * (i+1) +0.2*w
			y = h2 * (1 - self.cellRoundRate[i+1]) + 0.1*h
			y_avg = h2 * (1 - average[0]) + 0.1*h
			y_std_p = h2 * (1 - std_plus[0]) + 0.1*h
			y_std_m = h2 * (1 - std_minus[0]) + 0.1*h
			len_point = w2*0.25
			self.figCanvasEx1.create_line(x_p, y_p, x, y, width=3, fill="#98fb98")
			self.figCanvasEx1.create_line(x_p, y_avg, x, y_avg, width=1, fill="#ff0000")
			self.figCanvasEx1.create_line(x_p, y_std_p, x_p +len_point, y_std_p, width=1, fill="#ff69b4")
			self.figCanvasEx1.create_line(x_p, y_std_m, x_p +len_point, y_std_m, width=1, fill="#ff69b4")

		self.figCanvasEx1.create_text(w*0.2, h*0.1,  text = '1.0 -', font = fonts, anchor = "e")
		self.figCanvasEx1.create_text(w*0.2, h*0.1+(h*0.45),  text = '0.5 -', font = fonts, anchor = "e")
		self.figCanvasEx1.create_text(w*0.2, h*0.95,  text = '0.0 -', font = fonts, anchor = "e")
		xxx = int(len(self.cellRoundRate)/10)
		print(xxx, len(self.cellRoundRate))
		if(xxx>0):
			zzzz = int(str(xxx)[0]) * (10 **(len(str(xxx))-1))
			i = 0
			while(zzzz*i<len(self.cellRoundRate)):
				x = w2*(zzzz*i)+0.2*w
				print(x, w2, zzzz)
				self.figCanvasEx1.create_line(x, h, x, h+font_size*0.25)
				self.figCanvasEx1.create_text(x, h+font_size*0.75,  text = str(zzzz*i), font = fonts, anchor = 'n')
				i += 1

		
		self.figCanvasEx1.create_text(w*0.1, h*0.1+(h*0.45)-font_size,  text = '円', font = fonts, anchor = "e")
		self.figCanvasEx1.create_text(w*0.1, h*0.1+(h*0.45),  text = '形', font = fonts, anchor = "e")
		self.figCanvasEx1.create_text(w*0.1, h*0.1+(h*0.45)+font_size,  text = '度', font = fonts, anchor = "e")
		self.figCanvasEx1.create_text(0.6*w, h+font_size*2.0, text = '時間(frame)', font = fonts, anchor = 'n')
		#self.figCanvasEx1.create_text(w*0.1, h*0.1+(h*0.45)+20,  text = '', font = fonts, anchor = "e")
		'''
		self.graphEx1.plot(self.cellTime, self.cellRoundRate)
		self.graphEx1.plot(self.cellTime, average)
		self.graphEx1.plot(self.cellTime, std_plus)
		self.graphEx1.plot(self.cellTime, std_minus)
		self.figCanvasEx1.draw()
		'''
		self.masterImage = 0
		
	def graphView2(self, event):
		if(self.graphview==0):
			self.graphView1()
		else:
			h, w = self.videoShape
			self.movieCanvas = tkinter.Canvas(self.mainFrame, width = w, height = h)
			self.movieCanvas.bind('<ButtonPress-1>', self.onClicked)
			self.movieCanvas.bind('<ButtonPress-2>', self.cellSave)
			self.movieCanvas.bind('<Motion>', self.onMotion)
			self.master.bind('<KeyPress>', self.keyPress)
			self.movieCanvas.place(x = w*0.1, y = h*0.1)
			self.movieCanvas.create_image(0, 0, image=self.movie_image, anchor='nw')
			self.rect = self.movieCanvas.create_rectangle(self.targetArea2[1], self.targetArea2[0], self.targetArea2[3], self.targetArea2[2], tag = 'rect')
			
			self.graphview -= 2
		self.graphview += 1
		
	def graphOnImage(self, event):
		if(self.masterImage == 0):
			h, w = 100, 100
			#self.masterImage = tkinter.Toplevel()
			#self.masterImage.protocol("WM_DELETE_WINDOW", self.closeMasterImage)
			#self.rootFrame = ttk.Frame(self.masterImage, width = w, height = h)
			#self.rootFrame.pack()
			#self.rootImageCanvas = tkinter.Canvas(self.rootFrame, width = w, height = h)
			#self.rootImageCanvas.pack()

	def graphOnImageDraw(self, event):
		w = self.videoShape[1]*0.95
		w2 = ((w*0.8))/(len(self.cellTime))
		x = event.x
		print(x)
		num = -1
		digit = len(str(self.videoLength))
		for i in range(len(self.cellTime)):
			print((-0.85+i)*(w2), (x-w*0.2), (0.15+i)*(w2))
			if((-0.85+i)*(w2)<=(x-w*0.2)<= (0.15+i)*(w2)):
				if(i != num):
					image = cv2.imread(self.cellContours_path+ '/contours_' +str(int(self.cellTime[i])).zfill(digit)+'.png')
					image= cv2.resize(image, dsize =None, fx=4, fy=4)
					self.cellFrameImage = self.makeTkImage(image, self.master)
					self.cellFrameCanvas.create_image(0.3*self.videoShape[1], 0.3*self.videoShape[0], image = self.cellFrameImage)
					self.label_data[1][0].delete(0, tkinter.END)
					self.label_data[1][0].insert(tkinter.END, '('+str(int(self.cellRange_list[i][0]))+','+str(int(self.cellRange_list[i][1])) +')')
					self.label_data[2][0].delete(0, tkinter.END)
					self.label_data[2][0].insert(tkinter.END, str(self.cellArea[i]))
					self.label_data[3][0].delete(0, tkinter.END)
					self.label_data[3][0].insert(tkinter.END, str(int(self.cellRound[i])))
					self.label_data[4][0].delete(0, tkinter.END)
					self.label_data[4][0].insert(tkinter.END, str(self.cellRoundRate[i])[:5])
					self.label_data[0][0].delete(0, tkinter.END)
					self.label_data[0][0].insert(tkinter.END, str(self.cellTime[i]))
					num = i
	def closeMasterImage(self):
		self.masterImage.destroy()
		self.masterImage = 0
		
	def __create_menu(self):
		self.menu = tkinter.Menu(self.root, tearoff=0, background="#111111", foreground="#eeeeee", activebackground="#000000", activeforeground="#ffffff")
		self.menu.add_command(label = "Save", command=self.cellSave, font=(self.default_fontfamily, self.default_fontsize))
		self.menu.add_command(label = "Propaty", command=self.processing_propaty, font=(self.default_fontfamily, self.default_fontsize))
		self.menu.add_command(label = "Paste", command=self.__on_paste, font=(self.default_fontfamily, self.default_fontsize))
		self.menu.add_command(label = "Delete", command=self.__on_delete, font=(self.default_fontfamily, self.default_fontsize))
		self.menu.add_separator()
		self.menu.add_command(label = "Select all", command=self.__on_select_all, font=(self.default_fontfamily, self.default_fontsize))
		
		
		
	def processing_propaty(self):
		h, w = self.videoShape
		
		self.masterPropaty = tkinter.Toplevel()
		self.masterPropaty.protocol("WM_DELETE_WINDOW", self.closeProcessingPropaty)
		self.rootPropaty = ttk.Frame(self.masterPropaty, width = w*1.5, height = h)
		self.rootPropaty.pack()
		self.canvasPropaty = tkinter.Canvas(self.rootPropaty, width = w, height = h)
		self.canvasPropaty.place(y = 0, x = 0)
		self.framePropaty = ttk.Frame(self.masterPropaty, width = w*0.5, height = h)
		self.framePropaty.place(y = 0, x = w)
		self.addButtonPropaty = ttk.Button(self.framePropaty, text = '+', command = self.add_propaty)
		#self.subButtonPropaty = ttk.Button(self.framePropaty, text = '-', command = self.sub_propaty)
		self.addButtonPropaty.place(y=0, x=0)
		#self.subButtonPropaty.pack()
		self.processButtonPropaty = []
		j = 0
		for i in self.processName_list:
			self.processButtonPropaty.append([ttk.Button(self.framePropaty, text = str(i[0]), command = self.add_propaty)])
			self.processButtonPropaty[j][0].place(y = j*30+30, x = 20)
			j += 1

		
		
	def add_propaty(self):
		h, w = self.videoShape
		self.masterAddPropaty = tkinter.Toplevel()
		self.masterAddPropaty.protocol('WM_DELETE_WINDOW', self.closeAddPropaty)
		self.rootAddPropaty = ttk.Frame(self.masterAddPropaty, width = w*1.5, height = h)
		self.rootAddPropaty.pack()
		self.canvasAddPropaty = tkinter.Canvas(self.rootAddPropaty, width = w, height = h)
		self.canvasAddPropaty.place(y= 0, x = 0)
		self.frameAddPropaty = ttk.Frame(self.rootAddPropaty, width = w*0.5, height = h)
		self.frameAddPropaty.place(y = 0, x = w)
		fonts = ("MSゴシック", "10")
		texts = ['Grayscale', 'Threshold', 'Opening', 'Explore', 'Components', 'Contours']
		#print(self.processName_list)
		#print(i[0] for i in self.processName_list)
		texts2 = list(i[0] for i in self.processName_list)
		#print(texts2)
		self.dataComboboxAddPropaty = tkinter.StringVar()
		self.comboboxAddPropaty = ttk.Combobox(self.frameAddPropaty, height=3, width = 20, state = 'readonly', textvariable = self.dataComboboxAddPropaty, values = texts)
		self.comboButtonPropaty = ttk.Button(self.frameAddPropaty, text = 'set', command = self.show_propaty)

		self.dataCombobox2AddPropaty = tkinter.StringVar()
		self.combobox2AddPropaty = ttk.Combobox(self.frameAddPropaty, height=3, width = 20, state = 'readonly', textvariable = self.dataCombobox2AddPropaty, values = texts2)
		self.comboButton2Propaty = ttk.Button(self.frameAddPropaty, text = 'set', command = self.show_propaty)
		
		self.menuPropaty = tkinter.Menu(self.rootAddPropaty, tearoff=0, background="#111111", foreground="#eeeeee", activebackground="#000000", activeforeground="#ffffff")
		self.menuPropaty.add_command(label = "Grayscale", command=self.cellSave, font=(self.default_fontfamily, self.default_fontsize))
		self.menuPropaty.add_command(label = "Threshold", command=self.cellSave, font=(self.default_fontfamily, self.default_fontsize))
		self.menuPropaty.add_command(label = "Opening", command=self.processing_propaty, font=(self.default_fontfamily, self.default_fontsize))
		self.menuPropaty.add_command(label = "Explore", command=self.__on_paste, font=(self.default_fontfamily, self.default_fontsize))
		self.menuPropaty.add_command(label = "Delete", command=self.__on_delete, font=(self.default_fontfamily, self.default_fontsize))
		
		
		#self.selectButtonAddPropaty = ttk.Button(self.frameAddPropaty, text = 'v', command = self.__do_popup2, width = 2)
		self.comboboxAddPropaty.place(y = 0, x = w*0.1)
		self.comboButtonPropaty.place(y = 0, x = w*0.3)
		
		self.combobox2AddPropaty.place(y = 0.2*h, x = w*0.1)
		self.comboButton2Propaty.place(y = 0.2*h, x = w*0.3)
		#self.selectButtonAddPropaty.place(y = 0, x = w*0.1+int(fonts[1])*10)
		self.endButtonAddPropaty = ttk.Button(self.frameAddPropaty, text = 'ok', command = self.end_add_propaty, width = 4)
		self.endButtonAddPropaty.place(y = h*0.9, x = w*0.4)
		print('a')

	def show_propaty(self):
		h, w = self.videoShape
		process = self.dataComboboxAddPropaty.get()
		print(process)
		if(process == "Grayscale"):
			#image = cv2.imread(self.videoImage_path + '/image_' +str(0).zfill(len(str(self.videoLength)))+'.png')
			image = self.processImage_list[-1][0]
			image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			self.movie_image_gray = self.makeTkImage(image_gray, self.master)
			self.canvasAddPropaty.create_image(0, 0, image=self.movie_image_gray, anchor='nw')
			self.processImage = image_gray
			self.processOption = []
			
		if(process == "Threshold"):
			image = self.processImage_list[-1][0]
			self.movie_image_propaty = self.makeTkImage(image, self.master)
			self.canvasAddPropaty.create_image(0, 0, image=self.movie_image_propaty, anchor='nw')
			self.threshold = DoubleVar()
			self.scalePropaty = ttk.Scale(self.frameAddPropaty, 
			variable = self.threshold, 
			orient= HORIZONTAL, 
			length = w*0.3,
			from_=0, 
			to = 255, 
			command = self.show_thresh)
			self.scalePropaty.place(y = 0.5*h, x = 0.1*w)
		
		if(process == "Components"):
			image = self.processImage_list[-1][0]
			image = np.array(image, dtype = 'u1')
			#contours = cv2.connectedComponentsWithStats(image)[1]
			area = cv2.connectedComponentsWithStats(image)[2]
			#area_sort = area[np.argsort(area[:, 4])[::-1]]
			self.movie_image_propaty = self.makeTkImage(image, self.master)
			self.canvasAddPropaty.create_image(0, 0, image=self.movie_image_propaty, anchor='nw')
			text = []
			for i in range(len(area)):
				text.append(str(i))
			self.componentsNum = StringVar()
			self.componentsCombobox =  ttk.Combobox(self.frameAddPropaty, height=3, width = 20, state = 'readonly', textvariable = self.componentsNum, values = text)
			self.componentsComboboxButton =  ttk.Button(self.frameAddPropaty, text = 'set', command = self.show_components)
			self.componentsCombobox.place(y = 0.5*h, x = 0.1*w)
			self.componentsComboboxButton.place(y = 0.5*h, x = 0.3*w)

		if(process == "Contours"):
			#image = cv2.imread(self.videoImage_path + '/image_' +str(0).zfill(len(str(self.videoLength)))+'.png')
			self.show_contours()


	def closeAddPropaty(self):
		self.masterAddPropaty.destroy()

	def closeProcessingPropaty(self):
		self.masterPropaty.destroy()
		
	def sub_propaty(self):
		print('a')
		
	def end_add_propaty(self):
		for i in self.processButtonPropaty:
			i[0].destroy
		self.processButtonPropaty = []
		#print(self.processButtonPropaty, "destroy")
		self.processImage_list.append([copy.deepcopy(self.processImage)])
		processName = self.dataComboboxAddPropaty.get()
		processNums = 0
		for i in self.processName_list:
			if(processName in i[0]):
				processNums += 1
		if(processNums>0):
			processName = processName+'-'+str(processNums)

		self.processName_list.append([processName])
		self.processOption_list.append([copy.deepcopy(self.processOption)])
		print(self.processOption_list)
		j = 0
		for i in self.processName_list:
			self.processButtonPropaty.append([ttk.Button(self.framePropaty, text = str(i[0]), command = self.add_propaty)])
			self.processButtonPropaty[j][0].place(y = j*30+30, x = 20)
			j += 1
		#print(self.processButtonPropaty, "append")
		
		self.closeAddPropaty()

	def show_thresh(self, thresh):
		image = self.processImage_list[-1][0]	
		thresh = int(self.threshold.get())
		a, b = np.shape(image)
		image2 = copy.deepcopy(image)
		for i in range(a):
			for j in range(b):
				if(image[i][j]<thresh):
					image2[i][j] = 0
				else:
					image2[i][j] = 255
		image2 = np.array(image2, dtype = 'u1')
		self.movie_image_thresh = self.makeTkImage(image2, self.master)
		self.canvasAddPropaty.create_image(0, 0, image=self.movie_image_thresh, anchor='nw')
		self.processImage = image2
		self.processOption = [thresh] 
		
	def show_components(self):
		num = int(self.componentsNum.get())
		print(num)
		image = self.processImage_list[-1][0]
		image = np.array(image, dtype = 'u1')
		contours = cv2.connectedComponentsWithStats(image)[1]
		area = cv2.connectedComponentsWithStats(image)[2]
		area_sort = area[np.argsort(area[:, 4])[::-1]]
		a, b = np.shape(image)
		image2 = np.zeros((a, b))
		count = 0
		for i in range(a):
			for j in range(b):
				if(area[contours[i][j]][4]==area_sort[num][4]):
					count+=1
					image2[i][j] = 255
		print(count)
		image2 = np.array(image2, dtype = 'u1')
		self.movie_image_components = self.makeTkImage(image2, self.master)
		self.canvasAddPropaty.create_image(0, 0, image=self.movie_image_components, anchor='nw')
		self.processImage = image2
		self.processOption = [num]

	def show_contours(self):
		contours = self.processImage_list[-1][0]
		contours = np.array(contours, dtype = 'u1')
		h, w = np.shape(contours)
		img_black = np.zeros((h, w))
		que = []
		freemanChain = [[-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1]]
		chainNum = -8
		chainPre = 0
		roundLength = 0
		flag = False
		for i in range(h):
			for j in range(w):
				if flag:
					break
				if(contours[i][j] != 0):
					first = [i, j]
					flag = True
		flag = False
		xx = first[1]
		yy = first[0]
		for i in range(h):
			for j in range(w):
				if flag:
					break
				flag2 = False
				for k in range(len(freemanChain)) :
					if(flag2):
						break
					y = yy + freemanChain[k + chainNum][0]
					x = xx + freemanChain[k + chainNum][1]
					if(x>=0 and x< w and y>= 0 and y< h):
						if(contours[y][x]!=0):
							flag2 = True
							yy = y
							xx = x
							chainNum = k + chainNum -3
							if(chainNum<-8):
								chainNum += 8
							if(chainNum>0):
								chainNum -= 8
							que.append([y, x])
							img_black[y][x] = 255
							#image_w[y+Y1][x+X1] = [0, 255, 0]
							#menseki -= 0.5
							if((-1*chainNum)%2 == 0):
								roundLength += 1
							else:
								roundLength += 1.406
							if([yy, xx]==first):
								flag = True
			
		image2 = np.array(img_black, dtype = 'u1')
		self.movie_image_contours = self.makeTkImage(image2, self.master)
		self.canvasAddPropaty.create_image(0, 0, image=self.movie_image_contours, anchor='nw')
		self.processImage = image2
		self.processOption = []
	def __on_cut(self):
		self.event_generate("<<Cut>>")


	def __on_copy(self):
		self.event_generate("<<Copy>>")


	def __on_paste(self):
		self.event_generate("<<Paste>>")


	def __on_delete(self):
		# from tkinter/constants.py
		#first = self.index(tk.SEL_FIRST)
		#last = self.index(tk.SEL_LAST)
		
		first = self.index("sel.first")
		last = self.index("sel.last")
		self.delete(first, last)


	def __on_select_all(self):
		self.select_range(0, "end")


	def __bind_event(self):
		self.movieCanvas.bind('<ButtonPress-2>', self.__do_popup)


	def __do_popup(self, e):
		try:
			self.menu.tk_popup(e.x_root, e.y_root)
		finally:
			self.menu.grab_release()

	def __do_popup2(self):
		h, w = self.videoShape
		try:
			self.menuPropaty.tk_popup(y = 10+1, x = int(w*1.1))
		finally:
			self.menuPropaty.grab_release()

Scribble().run()