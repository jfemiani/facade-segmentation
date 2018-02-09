
# make sure FEATURES are in scope...

default = Munch()
default.fill = 2 # 2 is a positive label
default.outline = 1  # 1 is unlabeled, for the border
default.radius = 1 # 3 pixel thick border

# Windows have a slightly thicker outline
settings = munchify(dict(zip([str(f) for f in FEATURES], [default]*len(FEATURES))))
settings.window.outline = 3
settings.window.radius  = 6

# Facades have a much thicker outline
settings.facade.outline = 3
settings.facade.radius = 10

#print munch.toYAML(settings)