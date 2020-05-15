##Pesos de uma determinada carteira de investimento
##Possuindo a quantidade de acoes e o preco medio pago
##

import matplotlib.pyplot as plt

brsr6 = {}
brsr6['preco'] = 23.7
brsr6['qte'] = 100
brsr6['valor_pago'] = brsr6['preco'] * brsr6['qte']
# print(brsr6)

csna3 = {}
csna3['preco'] = 9.95
csna3['qte'] = 100
csna3['valor_pago'] = csna3['preco'] * csna3['qte']
# print(csna3)

enat3 = {}
enat3['preco'] = 10.89
enat3['qte'] = 100
enat3['valor_pago'] = enat3['preco'] * enat3['qte']
# print(enat3)

ggbr4 = {}
ggbr4['preco'] = 14.95
ggbr4['qte'] = 100
ggbr4['valor_pago'] = ggbr4['preco'] * ggbr4['qte']
# print(ggbr4)

itsa4 = {}
itsa4['preco'] = 12.85
itsa4['qte'] = 100
itsa4['valor_pago'] = itsa4['preco'] * itsa4['qte']
# print(itsa4)

klbn4 = {}
klbn4['preco'] = 2.947
klbn4['qte'] = 400
klbn4['valor_pago'] = klbn4['preco'] * klbn4['qte']
# print(klbn4)

oibr4 = {}
oibr4['preco'] = 1.84
oibr4['qte'] = 700
oibr4['valor_pago'] = oibr4['preco'] * oibr4['qte']
# print(oibr4)

petr4 = {}
petr4['preco'] = 25.79
petr4['qte'] = 200
petr4['valor_pago'] = petr4['preco'] * petr4['qte']
# print(petr4)

stbp3 = {}
stbp3['preco'] = 6.65
stbp3['qte'] = 300
stbp3['valor_pago'] = stbp3['preco'] * stbp3['qte']
# print(stbp3)

tiet11 = {}
tiet11['preco'] = 12.53
tiet11['qte'] = 800
tiet11['valor_pago'] = tiet11['preco'] * tiet11['qte']
# print(tiet11)

tiet4 = {}
tiet4['preco'] = 24.7
tiet4['qte'] = 1000
tiet4['valor_pago'] = tiet4['preco'] * tiet4['qte']
# print(tiet4)

trpl4 = {}
trpl4['preco'] = 24.54
trpl4['qte'] = 100
trpl4['valor_pago'] = trpl4['preco'] * trpl4['qte']
# print(trpl4)

vvar3 = {}
vvar3['preco'] = 4.83
vvar3['qte'] = 300
vvar3['valor_pago'] = vvar3['preco'] * vvar3['qte']
# print(vvar3)

itub4 = {}
itub4['preco'] = 35.00
itub4['qte'] = 200
itub4['valor_pago'] = itub4['preco'] * itub4['qte']
# print(itub4)

valorTotal = brsr6['valor_pago'] + csna3['valor_pago'] + enat3['valor_pago'] + ggbr4['valor_pago'] + itsa4['valor_pago'] + klbn4['valor_pago'] + oibr4['valor_pago'] + \
             petr4['valor_pago'] + stbp3['valor_pago'] + tiet11['valor_pago'] + tiet4['valor_pago'] + trpl4['valor_pago'] + vvar3['valor_pago'] + itub4['valor_pago']
# print(valorTotal)

labels = ['brsr6', 'csna3', 'enat3', 'ggbr4', 'itsa4', 'klbn4', 'oibr4', 'petr4', 'stbp3', 'tiet11', 'tiet4', 'trpl4', 'vvar3', 'itub4']
sizes = [brsr6['valor_pago'], csna3['valor_pago'], enat3['valor_pago'], ggbr4['valor_pago'], itsa4['valor_pago'], klbn4['valor_pago'], oibr4['valor_pago'],
         petr4['valor_pago'], stbp3['valor_pago'], tiet11['valor_pago'], tiet4['valor_pago'], trpl4['valor_pago'], vvar3['valor_pago'], itub4['valor_pago']]
# explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

