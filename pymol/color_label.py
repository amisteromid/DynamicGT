for i in cmd.get_names():
	#pdb,chain,_ = i.split('_')
	cmd.extract('ligand',"not polymer")
	# Make more contrast with certain threshold
	cmd.alter(f"{i} and b > 0.5", "b = 1 - (1 - b) * 0.99")
	cmd.alter(f"{i} and b < 0.5001", "b = b * 0.99")
	
	cmd.spectrum("b", "density_white_red", f'{i}', minimum=0, maximum=1)
	
	#cmd.color('gray20', f'{i} and not chain {chain}')
	#cmd.extract(f"{i}_not_{chain}", f"{i} and not chain {chain}")
	#cmd.set('cartoon_transparency', 0.25, f"{i}_not_{chain}")
	#cmd.set('cartoon_transparency', 0.15, f"{i} and not chain {chain}")
	cmd.hide('all')
	cmd.show('sphere', 'polymer')#f'{i} and chain {chain}'
	cmd.color('black', 'not polymer')
	cmd.show('sticks', 'not polymer')
	cmd.orient()
