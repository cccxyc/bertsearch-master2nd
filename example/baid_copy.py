import requests  # 发送请求
from bs4 import BeautifulSoup  # 解析页面
import pandas as pd  # 存入csv数据
import os  # 判断文件存在
from time import sleep  # 等待间隔
import random  # 随机
import re  # 用正则表达式提取url

headers = {
	"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36",
	"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
	"Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
	"Connection": "keep-alive",
	"Accept-Encoding": "gzip, deflate",
	"Host": "www.baidu.com",
	# 需要更换Cookie
	"Cookie": "BIDUPSID=1923B8E9CDD59499CE364EED6E23B51C; PSTM=1608625856; __yjs_duid=1_2da34db625583103192b9d28e9efad481619509758400; BDUSS=VdsdU9BcmU5RU4xZHp5bFlSNGl2bUJBbGNwZUtGNDBiN1NQVWZjdXZTbXdIS05pSVFBQUFBJCQAAAAAAAAAAAEAAAC-FtqAv7S~tLChY2hyb20AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALCPe2Kwj3tiT; BDUSS_BFESS=VdsdU9BcmU5RU4xZHp5bFlSNGl2bUJBbGNwZUtGNDBiN1NQVWZjdXZTbXdIS05pSVFBQUFBJCQAAAAAAAAAAAEAAAC-FtqAv7S~tLChY2hyb20AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALCPe2Kwj3tiT; BD_UPN=12314753; MCITY=-257:; BAIDUID=297A167C63E184429DE674F9702FE555:FG=1; BDSFRCVID=sjkOJexroG0C3gbj6jUGhMN7OmKKHPoTDYLEOwXPsp3LGJLVgGlOEG0PtjOBDqu-oxCnogKK3gOTH4PF_2uxOjjg8UtVJeC6EG0Ptf8g0M5; H_BDCLCKID_SF=tbkJ_DPhJIt3fP36q45HMt00qxby26nvtGO9aJ5nQI5nhh3zXTo8K4-S0M7uWJQRtacvXfQFQUbmjRO206oay6O3LlO83h525aT3Kl0MLPbceIOn5DcYyx_YyMnMBMPe52OnaIbx3fAKftnOM46JehL3346-35543bRTLnLy5KJYMDFRDT-Kj5OXjHR-5-7ybCPXoCtMfIL_Hnur0bC2XUI8LNDH2lTPQG62BDThyPJs8qrGQMcAe5b-hRO7ttoyLJ5CBJvbbnnnqU_4b45J2ML1Db0LKjvMtg3t3Db2Mq6oepvoDPJc3Mv30-jdJJQOBKQB0KnGbUQkeq8CQft20b0EeMtjKjLEtRk8oK_XtIvffPFk-PI3BT0vXP6-hnjy3bRmob3t2DQ8flbdXfIbXPuUyP7T2h3RymJ4Kb5e0D-Mjpjw2xv4bU-jXPoxJpOJ2eOGKPjaHR7WEnrvbURvDP-g3-AJQU5dtjTO2bc_5KnlfMQ_bf--QfbQ0hOhqP-jBRIEoC0XtK_BhKvPKITD-tFO5eT22-usW4jC2hcHMPoosIJJDPJDM6Kq3fn3WxCfBKjiahnxaMbUoqRHXnJi0btQDPvxBf7p52vQ2h5TtUJMjl7tQbQhqt4bLUTyKMnitKv9-pP2LpQrh459XP68bTkA5bjZKxtq3mkjbPbDfn028DKuDjtBDTj-DaRabK6aKC5bL6rJabC3qnccXU6q2bDeQN3mJ-ba2JcB-l3SBlub8n6oyU7mhp0vWtv4WbbvLT7johRTWqR4ep31XUonDh83M4nG0notHCOO5brO5hvvhn3O3MAM0MKmDloOW-TB5bbPLUQF5l8-sq0x0bOte-bQXH_E5bj2qRFeoID23H; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; delPer=0; BD_CK_SAM=1; BAIDUID_BFESS=297A167C63E184429DE674F9702FE555:FG=1; BDSFRCVID_BFESS=sjkOJexroG0C3gbj6jUGhMN7OmKKHPoTDYLEOwXPsp3LGJLVgGlOEG0PtjOBDqu-oxCnogKK3gOTH4PF_2uxOjjg8UtVJeC6EG0Ptf8g0M5; H_BDCLCKID_SF_BFESS=tbkJ_DPhJIt3fP36q45HMt00qxby26nvtGO9aJ5nQI5nhh3zXTo8K4-S0M7uWJQRtacvXfQFQUbmjRO206oay6O3LlO83h525aT3Kl0MLPbceIOn5DcYyx_YyMnMBMPe52OnaIbx3fAKftnOM46JehL3346-35543bRTLnLy5KJYMDFRDT-Kj5OXjHR-5-7ybCPXoCtMfIL_Hnur0bC2XUI8LNDH2lTPQG62BDThyPJs8qrGQMcAe5b-hRO7ttoyLJ5CBJvbbnnnqU_4b45J2ML1Db0LKjvMtg3t3Db2Mq6oepvoDPJc3Mv30-jdJJQOBKQB0KnGbUQkeq8CQft20b0EeMtjKjLEtRk8oK_XtIvffPFk-PI3BT0vXP6-hnjy3bRmob3t2DQ8flbdXfIbXPuUyP7T2h3RymJ4Kb5e0D-Mjpjw2xv4bU-jXPoxJpOJ2eOGKPjaHR7WEnrvbURvDP-g3-AJQU5dtjTO2bc_5KnlfMQ_bf--QfbQ0hOhqP-jBRIEoC0XtK_BhKvPKITD-tFO5eT22-usW4jC2hcHMPoosIJJDPJDM6Kq3fn3WxCfBKjiahnxaMbUoqRHXnJi0btQDPvxBf7p52vQ2h5TtUJMjl7tQbQhqt4bLUTyKMnitKv9-pP2LpQrh459XP68bTkA5bjZKxtq3mkjbPbDfn028DKuDjtBDTj-DaRabK6aKC5bL6rJabC3qnccXU6q2bDeQN3mJ-ba2JcB-l3SBlub8n6oyU7mhp0vWtv4WbbvLT7johRTWqR4ep31XUonDh83M4nG0notHCOO5brO5hvvhn3O3MAM0MKmDloOW-TB5bbPLUQF5l8-sq0x0bOte-bQXH_E5bj2qRFeoID23H; ZFY=jmuUX4xBrdUaSHtJ7rW9PqY9y3:B5uYi6VTNIZJrZWtY:C; PSINO=6; ZD_ENTRY=baidu; BD_HOME=1; BA_HECTOR=8k2ka02081012g0h2l0024881hripsm1l; B64_BOT=1; H_PS_PSSID=37780_36546_37972_37647_37554_37907_38012_36920_38034_37990_37932_38041_26350_37881; COOKIE_SESSION=6995_0_7_7_16_16_1_0_7_6_2_2_100742_0_1282_0_1673103352_0_1673102070|9#602485_12_1671002350|9; ab_sr=1.0.1_OTliZjI3OTVmOTVlNGNjOTU0ODIxYTFkNjRhOTYxMjRkZjllZTZjOWE0ZjE3MTc0ODk5MDEyOTc5OGJmOTE2N2M3OTNkMzI4YWE0MWUxZjIyNWNiMzU3MGM4ZGZlYzQxOGZkOWQ4MWRkZTFmZjJhM2I3OGNiMDlmYWY0OTRkZjMxZDUwOTJhMTMwMWJjMTI1ZjYzYTQxYjRjNTZiOWU0N2U1YTRiMDc4YzJlOGU4NTY4NDY3NDVhMDM0ZGQwMjBm; sugstore=1; H_PS_645EC=ab19RGSH/uSCy4V9J8YdI5+iaeHNR4nXzBRFN9B3A+MqtDDsh76533h3n0U; BDSVRTM=0"
}
v_keyword = "头痛怎么办？"
title_list = []

def get_real_url(v_url):
	"""
	获取百度链接真实地址
	:param v_url: 百度链接地址
	:return: 真实地址
	"""
	r = requests.get(v_url, headers=headers, allow_redirects=False)  # 不允许重定向
	if r.status_code == 302:  # 如果返回302，就从响应头获取真实地址
		real_url = r.headers.get('Location')
	else:  # 否则从返回内容中用正则表达式提取出来真实地址
		real_url = re.findall("URL='(.*?)'", r.text)[0]
	print('real_url is:', real_url)
	return real_url


for page in range(1):
	print('开始爬取第{}页'.format(page + 1))
	wait_seconds = random.uniform(1, 2)  # 等待时长秒
	print('开始等待{}秒'.format(wait_seconds))
	sleep(wait_seconds)  # 随机等待

	url = 'https://www.baidu.com/s?wd=' + v_keyword + '&pn=0&oq=' + v_keyword

	r = requests.get(url, headers=headers)
	html = r.text

	print('响应码是:{}'.format(r.status_code))
	soup = BeautifulSoup(html, 'html.parser')
	result_list = soup.find_all(class_='c-container')
	print('正在爬取:{},共查询到{}个结果'.format(url, len(result_list)))
	kw_list = []

	for result in result_list:
		title = result.find('a').text

		n_title = title.split(' ')[0]+'\n'
		print(title)
		if not title_list.__contains__(n_title):

			title_list.append(n_title)
Note=open('./eval_txt/'+v_keyword+'.txt',mode='w',encoding='utf-8')
print(title_list)
for a in title_list:
	Note.write(a)