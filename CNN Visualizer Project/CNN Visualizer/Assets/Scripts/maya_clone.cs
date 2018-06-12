//File is from https://github.com/keithlegg/Unity-3D-Maya-Navigation-/blob/master/maya.cs

//THIS IS A CSHARP REWRITE OF 
//http://danielskovli.com/folio/development/csharp/#maya-navigation-for-unity




using UnityEngine;
using System.Collections;
//using System.IO; //if you write files 


/// <summary>
/// Class for enabling maya-style camera control in the viewport.
/// </summary>
public class maya_clone : MonoBehaviour
{
    public float zoomSpeed = 1.2f;
    public float moveSpeed = .1f;
    public float rotateSpeed = 20.0f;
    public Vector3 startpos = new Vector3(0, 0, 0);

    public bool requireAltPressed = true;

    private GameObject orbitVector;
    private Quaternion orbt_rot_original;

    private Vector3 orbt_xform_original;


    // Use this for initialization
    void Start()
    {
        // Create a capsule (which will be the lookAt target and global orbit vector)
        orbitVector = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        orbitVector.transform.position = Vector3.zero;

        // Snap the camera to align with the grid in set starting position (otherwise everything gets a bit wonky)
        //transform.position = startpos;
        transform.LookAt(orbitVector.transform.position, Vector3.up);
        orbitVector.GetComponent<Renderer>().enabled = false; //hide the capsule object 	

        ///
        orbt_xform_original = orbitVector.transform.position;
        orbt_rot_original = orbitVector.transform.rotation;

        /*
		//FILE WRITING STUFF
	   var file = "C:/data/ORI.txt";
	
	   if ( File.Exists( file) ){
		  Debug.Log(file+" already exists.");
	   }	
	   */

    }


    /******************/

    void reset_xforms()
    {
        transform.parent = orbitVector.transform;
        orbitVector.transform.position = orbt_xform_original;
        orbitVector.transform.rotation = orbt_rot_original;
        transform.parent = null;
        transform.position = startpos;
    }


    /******************/

    void LateUpdate()
    {

        var x = Input.GetAxis("Mouse X");
        var y = Input.GetAxis("Mouse Y");


        if (Input.GetKey(KeyCode.Home))
        {
            this.reset_xforms();
        }
        /***********************/

        var wheelie = Input.GetAxis("Mouse ScrollWheel");

        if (wheelie < 0) // back
        {
            var currentZoomSpeed = 10f;
            transform.Translate(Vector3.forward * (wheelie * currentZoomSpeed));

        }
        if (wheelie > 0) // back
        {
            var currentZoomSpeed = 10f;
            transform.Translate(Vector3.forward * (wheelie * currentZoomSpeed));

        }

        /***********************/

        //Input.GetAxis("Mouse ScrollWheel") < 0) // back
        if (Input.GetKey(KeyCode.RightAlt) || Input.GetKey(KeyCode.LeftAlt) || !requireAltPressed)
        {

            // Distance between camera and orbitVector. We'll need this in a few places
            var distanceToOrbit = Vector3.Distance(transform.position, orbitVector.transform.position);

            //RMB - ZOOM
            if (Input.GetMouseButton(1))
            {

                // Refine the rotateSpeed based on distance to orbitVector
                var currentZoomSpeed = Mathf.Clamp(zoomSpeed * (distanceToOrbit / 50), 0.1f, 2.0f);

                // Move the camera in/out
                transform.Translate(Vector3.forward * (x * currentZoomSpeed));

                // If about to collide with the orbitVector, repulse the orbitVector slightly to keep it in front of us
                if (Vector3.Distance(transform.position, orbitVector.transform.position) < 3)
                {
                    orbitVector.transform.Translate(Vector3.forward, transform);
                }


                //LMB - PIVOT
            }
            else if (Input.GetMouseButton(0))
            {

                // Refine the rotateSpeed based on distance to orbitVector
                var currentRotateSpeed = Mathf.Clamp(rotateSpeed * (distanceToOrbit / 50), 1.0f, rotateSpeed);


                // Temporarily parent the camera to orbitVector and rotate orbitVector as desired
                transform.parent = orbitVector.transform;
                orbitVector.transform.Rotate(Vector3.right * (y * currentRotateSpeed));
                orbitVector.transform.Rotate(Vector3.up * (x * currentRotateSpeed), Space.World);
                transform.parent = null;


                //MMB - PAN
            }
            else if (Input.GetMouseButton(2))
            {

                // Calculate move speed
                var translateX = Vector3.right * (x * moveSpeed) * -1;
                var translateY = Vector3.up * (y * moveSpeed) * -1;

                // Move the camera
                transform.Translate(translateX);
                transform.Translate(translateY);

                // Move the orbitVector with the same values, along the camera's axes. In effect causing it to behave as if temporarily parented.
                orbitVector.transform.Translate(translateX, transform);
                orbitVector.transform.Translate(translateY, transform);
            }


            /****/

            //test to record camera's position
            //var foo = ( transform.position.ToString() +"\n");
            //System.IO.File.AppendAllText("C:/data/ORI.txt",foo );

        }//alt keys 



    }//lateupdate 

}