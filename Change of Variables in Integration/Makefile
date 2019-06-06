NAME := change\ of\ variables\ in\ integration

default:
	pdflatex --file-line-error $(NAME).tex
	
clean:
	rm $(NAME).aux
	rm $(NAME).log

spellcheck:
	aspell -c --mode=tex $(NAME).tex