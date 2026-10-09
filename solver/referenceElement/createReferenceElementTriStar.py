from referenceElement.createReferenceElementTri import create_reference_element_tri
from referenceElement.evaluateNodalBasisTri import evaluate_nodal_basis_tri

def create_reference_element_tri_star(reference_element):
    """
    Create the reference element with degree k+1 and compute geometry-related
    shape functions using degree k basis over the k+1 integration points.
    """

    # Create reference element of degree k+1
    reference_element_star = create_reference_element_tri(reference_element['degree'] + 1)

    # Evaluate basis functions of degree k at integration points of degree k+1
    NGeo, dNGeodxi, dNGeodeta = evaluate_nodal_basis_tri(
        reference_element_star['IPcoordinates'],
        reference_element['NodesCoord'],
        reference_element['degree']
    )

    # Assign geometric quantities to the enriched reference element
    reference_element_star['NGeo'] = NGeo
    reference_element_star['dNGeodxi'] = dNGeodxi
    reference_element_star['dNGeodeta'] = dNGeodeta
    reference_element_star['NodesCoordGeo'] = reference_element['NodesCoord']

    return reference_element_star
 
