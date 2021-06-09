====================
Community Guidelines
====================

**Docstring Standards**
=======================

Decimal Places
##############
Round to 2 decimal places in the input and output for marker positions

| Example: 
|     >>> C7 = np.array([256.78, 371.28, 1459.70])
|     ...
|     >>> np.around(findL5_Thorax(frame), 2)
|     array([265.16, 359.12, 1049.06])

Matrices with parenthesis
#########################
For matrix dimensions, surround it with parenthesis

| Example:
|     Returns the (x, y, z) marker positions of the midHip, a (1x3) array and L5, a (1x3) array in a tuple.

Backticks for variables
#######################
Add backticks (`) before and after variable names

| Example:
|     Returns the first value in \`p\` scaled by \`x\`, added by the second value in \`p\`.

**Math Standards**
==================

Marker Names
############
Marker names should follow the format: bolded m with the marker name as a subscript

| Example:
|     Marker name for RASI would be: **m**:sub:`RASI`

**Abbreviations**
=================
- `L`: left
- `R`: right
- `O`: origin
- `JC`: Joint Center
