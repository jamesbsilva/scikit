package scikit.graphics.dim2;

/**
* @(#)  GridCL
*/

import static scikit.util.Utilities.format;

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;
import javax.media.opengl.GL2;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLContext;
import javax.media.opengl.GLProfile;
import javax.media.opengl.awt.GLJPanel;
import javax.media.opengl.glu.gl2.GLUgl2;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JMenuItem;

import scikit.graphics.ColorChooser;
import scikit.graphics.ColorGradient;
import scikit.graphics.Drawable;
import scikit.opencl.BasicEventListenerCLGL;
import scikit.opencl.CLHelper;
import scikit.opencl.Scene2DCL;
import scikit.opencl.BasicUSI;
import scikit.util.Bounds;
import scikit.util.DoubleArray;
import scikit.util.FileUtil;

/**
*       GridCL is a lattice creating class with OpenGL/OpenCL interoperability
*   useful as a grid using OpenCL to update the state of the grid.
* 
* <br>
* 
* @author      James B. Silva <jbsilva @ bu.edu>                 
* @since       2013-07
*/
public class GridCL extends Scene2DCL {
        private final GLUgl2 glu = new GLUgl2();
	private final BasicUSI usi;
        private ColorChooser _colors = new ColorGradient();
	private BufferedImage _image = null;
	private int _w = 600, _h = 600;
	private int maxFrameLength = 800;
        private int latticeGeometry = 6;
        private int Lx;private int Ly;
	private double scaleLat = 0.045;
        private double[] _data = null;
        private int[] _pixelArray = null;
        private boolean _autoScale = true;
        private boolean _drawRange = false;
        private boolean clReady = false;
        private double _lo = 0, _hi = 1;
        private String drawLatticeKernel = "createLattice2D";
        private int posGlInd = 0;
        private int colGlInd = 0;
        private int posBuffNum = 0;
        private int colBuffNum = 1;
        private String defaultDeviceType = "GPU";
        private GLJPanel _canvas;        
        private GLContext _context;        
        private JFrame _frame;
        
    /**
    *         gridCL constructor.
    * 
    *  @param title - title of grid
    *  @param clin - clhelper in which to setup the grid
    *  @param Lxin - x coordinate size of grid
    *  @param Lyin - y coordinate size of grid
    *  @param G - geometry of lattice to draw (6 - triangular / 3- Honeycomb/ else square)
    *               use square for basic grid
    */ 
    public GridCL(String title,CLHelper clin, int Lxin, int Lyin, int G) {
            super(title,clin);
            initDrawKernel = drawLatticeKernel;
            initDrawKernelPosBufferNum = posBuffNum;
            initDrawKernelColBufferNum = colBuffNum;
            Lx = Lxin;
            Ly = Lyin;
            latticeGeometry = G;
            makeEventListener();
            usi = new BasicUSI();
            //initDrawLattice(Lx,0,1);
            if(_component == null)makeFrame(_canvas);            
        }
        // make the GLEventListener for this grid
        private void makeEventListener(){
            System.out.println("Grid Dimensions | Lx: "+Lx+"  Ly: "+Ly+"   width: "+_w+"   height: "+_h);
            _bel = new BasicEventListenerCLGL(null, 4*Lx*Ly,4*Lx*Ly,this,_w,_h);
            clscheduler = _bel.getCLScheduler();
            _canvas.addGLEventListener(_bel);
        }
        /**
        *         isOpenCLReady returns true if CLHelper is properly
        *   setup for interoperability. Any calls with CLHelper on a CLGL context
        *   before this will not cause good behavior.
        * 
        *  @return true - if CLGL is properly initiated
        */ 
        public boolean isOpenCLReady(){
            return _bel.getCLGLInitStatus();
        }
        
	public void clear() {
            // remove data first because super.clear() will cause a drawAll() operation
            _w = _h = 0;
            _image = null;
            _data = null;
            super.clear();
	}

	public void setColors(ColorChooser colors) {
            _colors = colors;
	}
	
	public void setAutoScale() {
            _autoScale = true;
	}
	
	public void setScale(double lo, double hi) {
            _autoScale = false;
            _lo = lo;
            _hi = hi;
	}
	
	public void setDrawRange(boolean b) {
            _drawRange = b;
	}
	
        private void assertGridDim(int w, int h){
            if(w != Lx || h != Ly){
                throw new IllegalArgumentException("Data Array length " + (w*h)
                                + " does not fit specified grid shape (" + Lx + "*" + Ly + ")");

            }
        }
        
        /**
        *         getEventListener returns the GLEventListener derived from 
        *   BasicEventListenerCL which will be used in this grid.
        * 
        *  @return GLEventListener being used for grid
        */ 
        public BasicEventListenerCLGL getEventListener(){
            return _bel;
        }
        
        /**
        *         getCLHelperGrid returns the CLHelper used by this grid
        * 
        *  @return CLHelper used by this grid
        */ 
        public CLHelper getCLHelperGrid(){
            return clhelper;
        }
                
	// Override getImage() to return the "native" pixel-map image
	public BufferedImage getImage(int width, int height) {
		return _image;
	}
	
	public BufferedImage getImage() {
		return _image;
	}
       
	@Override
        protected JComponent createCanvas() {
            //return GLJPanel canvas
            GLJPanel canvas = basicInitGL();
            return canvas;
	}
        
	private void makeFrame(GLJPanel canvas){
            System.out.println("GridCL |  Making Frame");
            JFrame frame = new JFrame("GridCL | "+getTitle());
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.add(canvas);
            frame.setSize(_w, _h);
            frame.setVisible(true);
            _frame = frame;
        }
        
        // basic OpenGL init 
        private GLJPanel basicInitGL(){
            GLCapabilities config = new GLCapabilities(GLProfile.get(GLProfile.GL2));
            config.setSampleBuffers(true);
            config.setNumSamples(4);            
            System.out.println("GridCL | Init GL Canvas");
            GLJPanel canvas = new GLJPanel(config);
            _canvas = canvas;
            return canvas;
        }
        
        /**
        *         initCLGL initiates the JOCL/JOGL in the CLHelper to be used.
        * 
        */ 
        @Override
        public void initCLGL(GLAutoDrawable drawable){
            _context = drawable.getContext();
            System.out.println("GridCL | Initializing OpenCL");
            //System.out.println("GL Info:   Context: "+drawable.getContext());
            clhelper.initializeOpenCL(defaultDeviceType, true, drawable.getContext());
            posGlInd = _bel.getPosGLind();
            colGlInd = _bel.getColGLind();
            //clearMouseListeners();
            usi.init(_canvas);
            _bel.setCLhelper(clhelper);
            initLat();
            System.out.println("GridCL | Done Init Pre Buff");
        }
        
        private void clearMouseListeners(){
            MouseListener[] mls = _canvas.getMouseListeners();
            for(int u = 0; u < mls.length;u++){
                _canvas.removeMouseListener(mls[u]);
            }
        }
        // initial drawing of lattice
        private void initLat(){
            int posBuffSize = 4*Lx*Ly;
            int colBuffSize = 4*Lx*Ly;
            float[] din = new float[posBuffSize];
            float[] din2 = new float[colBuffSize];
            clhelper.createKernelFromSource(initDrawKernel,getDrawKernelAsString());
            clhelper.createFloatBufferGL(initDrawKernel, initDrawKernelPosBufferNum, posBuffSize, din, "rw", true, _bel.getPosGLind());
            clhelper.createFloatBufferGL(initDrawKernel, initDrawKernelColBufferNum,  colBuffSize, din2, "rw", true, _bel.getColGLind());    
            clhelper.setIntArg(initDrawKernel,0, (int)(Math.sqrt(posBuffSize/4)));
            clhelper.setIntArg(initDrawKernel,1, latticeGeometry);
            clhelper.setFloatArg(initDrawKernel,0,(float)scaleLat);
            clhelper.setKernelArg(initDrawKernel);
            clhelper.runKernel(initDrawKernel, (posBuffSize/4),1);    
        }
        
	@Override
        protected void drawBackground(Gfx2D g) {
            // looks better without background
	}
	
	@Override
        protected List<Drawable<Gfx2D>> getAllDrawables() {
		List<Drawable<Gfx2D>> ds = new ArrayList<Drawable<Gfx2D>>();
		ds.add(_gridDrawable);
		ds.addAll(super.getAllDrawables());
		return ds;
	}
	
	@Override
        protected List<JMenuItem> getAllPopupMenuItems() {
		List<JMenuItem> ret = new ArrayList<JMenuItem>(super.getAllPopupMenuItems());

		JMenuItem menuItem = new JCheckBoxMenuItem("Display range", _drawRange);
		menuItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				_drawRange = !_drawRange;
				animate();
			}
		});
		ret.add(menuItem);	
		if (_data != null) {
			menuItem = new JMenuItem("Save grid data ...");
			menuItem.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					saveData("grid.txt");
				}
			});
			ret.add(menuItem);
			menuItem = new JMenuItem("Save image ...");
			menuItem.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					saveImage("grid.png");
				}
			});
			ret.add(menuItem);
		}
		return ret;
	}
	
	private void findRange() {
            if (_autoScale) {
                _lo = DoubleArray.min(_data);
                _hi = DoubleArray.max(_data);
            }
	}
	
	
	private Drawable<Gfx2D> _gridDrawable = new Drawable<Gfx2D>() {
            public void draw(Gfx2D g) {
            if (_image != null) {
                ((Gfx2DSwing)g).drawImage(_image, 0, 0, 1, 1);
                if (_drawRange) {
                    g.setProjection(g.pixelBounds()); // draw strings at fixed pixels
                    String str1 = "lo = "+format(_lo);
                    String str2 = "hi = "+format(_hi);
                    double border = 4;
                    double offset = 4;
                    double w = Math.max(g.stringWidth(str1), g.stringWidth(str2));
                    double h = g.stringHeight("");
                    g.setColor(new Color(1f, 1f, 1f, 0.5f));
                    g.fillRect(offset, offset, 2*border+w, 3*border+2*h);
                    g.setColor(Color.BLACK);
                    g.drawString(str1, offset+border, offset+h+2*border);
                    g.drawString(str2, offset+border, offset+border);
                    g.setProjection(g.viewBounds()); // return to scene coordinates
                }
            }
            }
            public Bounds getBounds() {
                return new Bounds(0, 1, 0, 1);
            }
	};
      
	private void saveData(String fname) {
            try {
                fname = FileUtil.saveDialog(_component, fname);
                if (fname != null) {
                    PrintWriter pw = FileUtil.pwFromString(fname);
                    FileUtil.writeOctaveGrid(pw, _data, _w, 1);
                    pw.close();
                }
            } catch (IOException e) {}
	}

	private void saveImage(String fname) {
            try {
                fname = FileUtil.saveDialog(_component, fname);
                if (fname != null) {
                        ImageIO.write(_image, "png", new File(fname));
                }
            } catch (IOException e) {}
	}
        /**
        *         interact is an interface method with UI like the BasicUSI and OpenGL. 
        * 
        *   @param gl - GL2 obj
        */
        public void interact(GL2 gl){
           usi.interact(gl);
        }
        
        @Override
        protected void drawAll(Object g) {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
        
        private String getDrawKernelAsString(){
            return 
                "float getXrCentered(int u,int v,int Geo,int L);\n" +
                "float getYrCentered(int u,int v,int Geo,int L);\n" +
                "float getXr(int u,int v,int Geo);\n" +
                "float getYr(int u,int v,int Geo);\n" +
                "float getXcoordTri(int i, int j);\n" +
                "float getYcoordTri(int i, int j);\n" +
                "float getXcoordHoney(int i, int j);\n" +
                "float getYcoordHoney(int i, int j);\n" +
                "\n" +
                "__kernel void createLattice2D(__global float4 * latPos,__global float4 *latCol, "
                        + "int L, int geo, float scale){\n" +
                "    unsigned int i = get_global_id(0);\n" +
                "\n" +
                "    // calculate uv coordinates\n" +
                "    int u = (i%L);\n" +
                "    int v = ((i/L)%L);\n" +
                "    float x = getXrCentered(u,v,geo,L)/(L*scale);\n" +
                "    float y = getYrCentered(u,v,geo,L)/(L*scale);\n" +
                "\n" +
                "    // write output vertex\n" +
                "    latPos[u+v*L] = (float4)(x, 0.0f, y, 1.0f);\n" +
                "    latCol[u+v*L] = (float4)(0.0f, 1.0f, 1.0f, 255.0f);\n" +
                "    return;\n" +
                "}\n" +
                "\n" +
                "float getXrCentered(int u,int v,int Geo,int L){\n" +
                "    int cenInd = (int)(L/2.0);\n" +
                "    float xcen = getXr(cenInd,cenInd,Geo);\n" +
                "    return (getXr(u,v,Geo)-xcen);\n" +
                "}\n" +
                "\n" +
                "float getYrCentered(int u,int v,int Geo, int L){\n" +
                "    int cenInd = (int)(L/2.0);\n" +
                "    float ycen = getYr(cenInd,cenInd,Geo);\n" +
                "    return (getYr(u,v,Geo)-ycen);\n" +
                "}\n" +
                "\n" +
                "\n" +
                "float getXr(int u,int v,int Geo){\n" +
                "    if(Geo ==  6){\n" +
                "        return getXcoordTri(u,v); \n" +
                "    }else if(Geo ==  3){\n" +
                "        return getXcoordHoney(u,v); \n" +
                "    }else{\n" +
                "        return u;\n" +
                "    }\n" +
                "}\n" +
                "\n" +
                "float getYr(int u,int v, int Geo){\n" +
                "    if(Geo ==  6){\n" +
                "        return getYcoordTri(u,v); \n" +
                "    }else if(Geo ==  3){\n" +
                "        return getYcoordHoney(u,v); \n" +
                "    }else{\n" +
                "        return v;\n" +
                "    }\n" +
                "}\n" +
                "\n" +
                "float getXcoordTri(int i, int j){\n" +
                "    return (j%2 == 0) ? (float)i : (float)i + 0.5;\n" +
                "}\n" +
                "\n" +
                "float getYcoordTri(int i, int j){\n" +
                "    return ((float)j)*sqrt(3.0f)/2.0;\n" +
                "}\n" +
                "\n" +
                "float getXcoordHoney(int i, int j){\n" +
                "    float offset = (j%2 == 0) ? floor((float)(i/2)) : (floor((float)(i/2))+1.5);\n" +
                "    return (((float)i))+offset;\n" +
                "}\n" +
                "\n" +
                "float getYcoordHoney(int i, int j){\n" +
                "    return ((float)j)*sqrt(3.0f)/2.0;\n" +
                "}\n";
        }
        
}
