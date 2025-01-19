BEGIN;

CREATE TABLE
  Lecturers (
    LecturerId int NOT NULL,
    LecturerName varchar(100) NOT NULL,
    PRIMARY KEY (LecturerId)
  );

CREATE TABLE
  Courses (
    CourseId int NOT NULL,
    CourseName varchar(100) NOT NULL,
    PRIMARY KEY (CourseId)
  );

CREATE TABLE
  Groups (
    GroupId int NOT NULL,
    GroupName char(6) NOT NULL,
    PRIMARY KEY (GroupId),
    CONSTRAINT UniqueGroupName UNIQUE (GroupName)
  );

CREATE TABLE
  Students (
    StudentId int NOT NULL,
    StudentName varchar(100) NOT NULL,
    GroupId int NOT NULL,
    PRIMARY KEY (StudentId),
    CONSTRAINT StudentsGroupFK FOREIGN KEY (GroupId) REFERENCES Groups (GroupId)
  );

CREATE TABLE
  Marks (
    StudentId int NOT NULL,
    CourseId int NOT NULL,
    Mark int NOT NULL,
    PRIMARY KEY (StudentId, CourseId),
    CONSTRAINT MarksStudentsFK FOREIGN KEY (StudentId) REFERENCES Students (StudentId),
    CONSTRAINT MarksCoursesFK FOREIGN KEY (CourseId) REFERENCES Courses (CourseId)
  );

CREATE TABLE
  Plan (
    CourseId int NOT NULL,
    GroupId int NOT NULL,
    LecturerId int NOT NULL,
    PRIMARY KEY (CourseId, GroupId),
    CONSTRAINT ActualLecturersCoursesFK FOREIGN KEY (CourseId) REFERENCES Courses (CourseId),
    CONSTRAINT ActualLecturersGroupsFK FOREIGN KEY (GroupId) REFERENCES Groups (GroupId),
    CONSTRAINT ActualLecturersLecturersFK FOREIGN KEY (LecturerId) REFERENCES Lecturers (LecturerId)
  );

END;
