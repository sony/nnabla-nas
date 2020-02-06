import json

from args import Configuration

conf = json.load(open('examples/tests.json'))
conf['search'] = True
conf = Configuration(conf)

t = conf.dataloader['train']
v = conf.dataloader['valid']


x, y = t.next()
xx, yy = v.next()

print(x.shape, y.shape)
print(xx.shape, yy.shape)
