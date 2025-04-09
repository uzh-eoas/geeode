#!/usr/bin/env python
import os
import numpy as np
from pylatex import (
    Alignat,
    Axis,
    Document,
    Figure,
    Math,
    Matrix,
    Plot,
    Section,
    Subsection,
    Tabular,
    TikZ,
    LargeText,
    PageStyle,
    Head,
    MiniPage,
    LineBreak,
    NewLine,
    Command,
    NoEscape,
    Itemize
)
from pylatex.utils import italic, bold
import matplotlib.pyplot as plt


# Appendix

np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = 2 * np.log(x + np.random.normal(0, 1, 100)) + 3

coords = [(x_, y_) for x_, y_ in zip(x.flatten(), y.flatten())]

geometry_options = {"tmargin": "1cm", "lmargin": "1cm", "rmargin": "1cm", "bmargin": "1cm", "paperheight":"90cm"}
doc = Document(geometry_options=geometry_options)
header = PageStyle("header")

with doc.create(MiniPage(align="c")):
    doc.append(LargeText(bold("DE Optimization Explained")))
    doc.append(LineBreak())

with doc.create(Section("Optimizing a Logarithmic Function", numbering = False)):

    with doc.create(Subsection("Example Data created from a Logarithmic Function", numbering = False)):
        doc.append(NoEscape('{'))
        doc.append(Command('centering'))
        with doc.create(TikZ()):
            plot_options = "height=12cm, width=12cm, grid=major"
            with doc.create(Axis(options=plot_options)) as plot:
                plot.append(Plot(name="Underlying Model", func="2 * ln(x + 1) + 3" , options=["smooth","thick","dotted"]))
                coordinates = coords
                plot.append(Plot(name="Data", coordinates=coordinates, options = ["only marks"]))
        doc.append(Command('par'))
        doc.append(NoEscape('}'))

    with doc.create(Subsection("Given the example data, fit a logarithmic function:", numbering = False)):
        with doc.create(Alignat(numbering=False, escape=False)) as agn:
            agn.append("y = a * ln(x + b) + c")
        doc.append("For a logarithmic equation in this form, optimize for 3 parameters:")
        with doc.create(Alignat(numbering=False, escape=False)) as agn:
            agn.append(r"(\ a\ ,\ b\ ,\ c\ )")

with doc.create(Section("Begin with an Initial Population of Models", numbering = True)):
    doc.append("The initial population is a matrix-like list of of randomly (or pseudo-randomly) generated vectors:")
    with doc.create(Alignat(numbering=False, escape=False)) as agn:
        agn.append(r"\mathbf{P} = (\ \mathit{v_1}\ , \\ \ \vdots\ \\ \ \mathit{v_n}\ )\\")
    doc.append("the element of each being parameters from the model being optimized:")
    with doc.create(Alignat(numbering=False, escape=False)) as agn:
        agn.append(r"\mathit{v_1} = (\ a_1\ ,\ b_1\ ,\ c_1) \\")
        agn.append(r"\vdots \ \ \ \vdots \ \ \ \ \ \\")
        agn.append(r"\mathit{v_n} = (\ a_n\ ,\ b_n\ ,\ c_n) \\")
    with doc.create(Subsection("Visualize", numbering = False)):
        doc.append(NoEscape('{'))
        doc.append(Command('centering'))
        with doc.create(TikZ()):
            plot_options = "height=12cm, width=12cm, grid=major"
            with doc.create(Axis(options=plot_options)) as plot:
                plot.append(Plot(name="Underlying Model", func="2 * ln(x + 1) + 3" , options=["smooth","thick","dotted"]))
                plot.append(Plot(name="Population Member 1", func="1.4 * ln(x + 0.5) + 1.7" , options=["smooth","thick","red"]))
                plot.append(Plot(name="Population Member 2", func="2.4 * ln(x + 1.5) + 4.2" , options=["smooth","thick","blue"]))
                plot.append(Plot(name="Population Member 3", func="1.9 * ln(x + 1.1) + 3.1" , options=["smooth","thick","green"]))
                coordinates = coords
                plot.append(Plot(name="Data", coordinates=coordinates, options = ["only marks"]))
        doc.append(Command('par'))
        doc.append(NoEscape('}'))

with doc.create(Section("Define a fitness function", numbering = True)):
    doc.append("As each member of the population defines a potential model (i.e., curve) to fit over the data, define a fitness function to compare each population member. For example, the root mean square error of each curve can be used:")
    with doc.create(Alignat(numbering=False, escape=False)) as agn:
        agn.append(r" RMSE = \sqrt{{\frac{\Sigma (y - p)^2}{n}}}")
        agn.append(r"\\")
        agn.append(r"\mathit{Fitness}(\mathit{v}) = RMSE(\mathit{v})")
    doc.append("where")
    with doc.create(Itemize()) as item_list:
        item_list.add_item(italic(r"y "))
        item_list.append(r" is the actual value of the data")
        item_list.add_item(italic("p "))
        item_list.append(r" is the predicted value from a model ")
        item_list.add_item(italic("n "))
        item_list.append(r" is the population size")

with doc.create(Section("Iterate the Population", numbering = True)):
    doc.append("The population must evolve across discrete steps, called iterations, with each iteration involving 2 procedures: Mutation and Crossover")
    with doc.create(Subsection("Mutation", numbering = True)):
        doc.append("Mutation involves combining population members to create a new potential mutant population member:")
        with doc.create(Alignat(numbering=False, escape=False)) as agn:
            agn.append(r"\mathit{v_m} = F(\mathit{v_x \ , \ldots \ , v_z})")
        doc.append(r"where ")
        doc.append(NoEscape(r"\textit{v\textsubscript{m}} "))
        doc.append(r"is the mutant vector/model, and ")
        doc.append(NoEscape(r"\textit{v\textsubscript{x}} "))
        doc.append(r" through ")
        doc.append(NoEscape(r"\textit{v\textsubscript{z}} "))
        doc.append(r" are any number of randomly drawn models from the population. ")
        doc.append("Mutation functions are often simple arithmetic combinations of candidate models. For example, the default mutation function in this implementation is:")
        with doc.create(Alignat(numbering=False, escape=False)) as agn:
            agn.append(r"\mathit{v_m} = \mathit{v_x} + M * (\mathit{v_y} - \mathit{v_z} )")
        doc.append(r"where ")
        doc.append(NoEscape(r"\textit{v\textsubscript{m}} "))
        doc.append(r"is the mutant vector, and ")
        doc.append(NoEscape(r"\textit{v\textsubscript{x}} "))
        doc.append(r" through ")
        doc.append(NoEscape(r"\textit{v\textsubscript{z}} "))
        doc.append(r" are randomly selected models from the population. Additionally, ")
        doc.append(italic("M "))
        doc.append(" is a simple numerical constant set by the user. Constants such as this one are often included in mutation functions as they allow users to influence variation within the mutation process.")
    with doc.create(Subsection("Crossover", numbering = True)):
        doc.append("Crossover involves selecting or creating a function that determines whether a mutant model replaces any single population member, as this function is applied to every individual population member. ")
        doc.append("A simple and commonly used crossover function is a simple binomial function wherein a randomly determined constant number (between 0 and 1) is compared to a crossover constant:")
        with doc.create(Alignat(numbering=False, escape=False)) as agn:
            agn.append(r"v_n = v_m\ if\ \mathit{\mathbf{X}} > C\ else\ v_n = v_i")
        doc.append(r"where ")
        doc.append(NoEscape(r"\textit{v\textsubscript{n}} "))
        doc.append(r"is the model being recorded during the iteration, ")
        doc.append(NoEscape(r"\textit{v\textsubscript{m}} "))
        doc.append(r" is the potential mutant model, ")
        doc.append(NoEscape(r"\textit{v\textsubscript{i}} "))
        doc.append(r" is the current model being considered, ")
        doc.append(NoEscape(r"\textit{\textbf{X}} "))
        doc.append(r" is a randomly generated constant between 0 and 1, and ")
        doc.append(NoEscape(r"\textit{C} "))
        doc.append(" is a simple numerical constant (set by the user) between 0 and 1. ")
        doc.append(" In this case, if a user sets a value for ")
        doc.append(NoEscape(r"\textit{C} "))
        doc.append(" that is closer to 0, more mutants substitutions will occur. If the user sets a value closer to 1, fewer substitutions will occur.")
with doc.create(Section("Finalize via the Fitness Function", numbering = True)):
    doc.append(r"Once the initial population of models has evolved, the final step is to compute the fitness value for each evolved model:")
    with doc.create(Alignat(numbering=False, escape=False)) as agn:
        agn.append(r"v_f_i_n_a_l = Fitness_B_e_s_t(\mathbf{P})")
    doc.append(r"The model with the best RMSE is the optimal solution outputted by the algorithm at that particular iteration. ")
    doc.append(r"Of note, the algorithm can be iterated any arbitrary number of times. It is best to stop when the algorithm converges to a model with a fitness function value that does not improve over further iterations.")
    with doc.create(Subsection("Optimal Solution", numbering = False)):
        doc.append(NoEscape('{'))
        doc.append(Command('centering'))
        with doc.create(TikZ()):
            plot_options = "height=12cm, width=12cm, grid=major"
            with doc.create(Axis(options=plot_options)) as plot:
                plot.append(Plot(name="Underlying Model", func="2 * ln(x + 1) + 3" , options=["smooth","thick","dotted"]))
                plot.append(Plot(name="Fittest Curve", func="1.9 * ln(x + 1.1) + 3.1" , options=["smooth","thick","green"]))
                coordinates = coords
                plot.append(Plot(name="Data", coordinates=coordinates, options = ["only marks"]))
        doc.append(Command('par'))
        doc.append(NoEscape('}'))

doc.preamble.append(header)
doc.change_document_style("header")
doc.generate_pdf("appendix", clean_tex=False, compiler='pdflatex')


# Main Figure

np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = 2 * np.log(x + np.random.normal(0, 1, 100)) + 3

coords = [(x_, y_) for x_, y_ in zip(x.flatten(), y.flatten())]
coords

figure_options = {"tmargin": "0.5cm", "lmargin": "0.5cm", "rmargin": "0.5cm", "bmargin": "0.5cm", "paperheight":"15cm", "paperwidth":"18cm"}
fig = Document(geometry_options=figure_options)

fig.append(Command('pagenumbering{gobble}'))
with fig.create(Subsection("", numbering = False)):
    fig.append(NoEscape('{'))
    fig.append(Command('centering'))
    with fig.create(TikZ()):
        plot_options = "height=12cm, width=12cm, grid=major"
        with fig.create(Axis(options=plot_options)) as plot:
            plot.append(Plot(name="Underlying Model", func="2 * ln(x + 1) + 3" , options=["smooth","thick","dotted"]))
            plot.append(Plot(name="Population Member 1", func="1.4 * ln(x + 0.5) + 1.7" , options=["smooth","thick","red"]))
            plot.append(Plot(name="Population Member 2", func="2.4 * ln(x + 1.5) + 4.2" , options=["smooth","thick","blue"]))
            plot.append(Plot(name="Population Member 3", func="1.9 * ln(x + 1.1) + 3.1" , options=["smooth","thick","green"]))
            coordinates = coords
            plot.append(Plot(name="Data", coordinates=coordinates, options = ["only marks"]))
    fig.append(Command('par'))
    fig.append(NoEscape('}'))


fig.generate_pdf("main_figure", clean_tex=False, compiler='pdflatex')
