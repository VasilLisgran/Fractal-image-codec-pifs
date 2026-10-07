from PIL import Image

im = Image.open('coffee_512.png')
for q in (95, 75, 50, 30):
    im.save(f'coffee_q{q}.jpg', quality=q)
    print('сохранён', f'coffee_q{q}.jpg')