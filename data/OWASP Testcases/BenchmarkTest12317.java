/**
* OWASP Benchmark Project v1.1
*
* This file is part of the Open Web Application Security Project (OWASP)
* Benchmark Project. For details, please see
* <a href="https://www.owasp.org/index.php/Benchmark">https://www.owasp.org/index.php/Benchmark</a>.
*
* The Benchmark is free software: you can redistribute it and/or modify it under the terms
* of the GNU General Public License as published by the Free Software Foundation, version 2.
*
* The Benchmark is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
* even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details
*
* @author Dave Wichers <a href="https://www.aspectsecurity.com">Aspect Security</a>
* @created 2015
*/

package org.owasp.benchmark.testcode;

import java.io.IOException;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@WebServlet("/BenchmarkTest12317")
public class BenchmarkTest12317 extends HttpServlet {
	
	private static final long serialVersionUID = 1L;
	
	@Override
	public void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		doPost(request, response);
	}

	@Override
	public void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
	
		String[] values = request.getParameterValues("foo");
		String param;
		if (values.length != 0)
		  param = request.getParameterValues("foo")[0];
		else param = null;

		String bar = new Test().doSomething(param);
		
		new java.io.File(new java.io.File(org.owasp.benchmark.helpers.Utils.testfileDir),bar);
	}  // end doPost

    private class Test {

        public String doSomething(String param) throws ServletException, IOException {

		// Chain a bunch of propagators in sequence
		String a60812 = param; //assign
		StringBuilder b60812 = new StringBuilder(a60812);  // stick in stringbuilder
		b60812.append(" SafeStuff"); // append some safe content
		b60812.replace(b60812.length()-"Chars".length(),b60812.length(),"Chars"); //replace some of the end content
		java.util.HashMap<String,Object> map60812 = new java.util.HashMap<String,Object>();
		map60812.put("key60812", b60812.toString()); // put in a collection
		String c60812 = (String)map60812.get("key60812"); // get it back out
		String d60812 = c60812.substring(0,c60812.length()-1); // extract most of it
		String e60812 = new String( new sun.misc.BASE64Decoder().decodeBuffer( 
		    new sun.misc.BASE64Encoder().encode( d60812.getBytes() ) )); // B64 encode and decode it
		String f60812 = e60812.split(" ")[0]; // split it on a space
		org.owasp.benchmark.helpers.ThingInterface thing = org.owasp.benchmark.helpers.ThingFactory.createThing();
		String bar = thing.doSomething(f60812); // reflection

            return bar;
        }
    } // end innerclass Test

} // end DataflowThruInnerClass
