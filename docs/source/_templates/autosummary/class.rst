{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: {{ ('Methods') }}

   .. autosummary::
       :toctree: .
       {% for item in methods %}
          {%- if item == "__init__" %}
          {%- else %}
            ~{{ name }}.{{ item }}
          {%- endif %}
       {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ ('Attributes') }}

   .. autosummary::
       {% for item in attributes %}
          ~{{ name }}.{{ item }}
       {%- endfor %}
   {% endif %}
   {% endblock %}